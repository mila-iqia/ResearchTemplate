from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable

import gym
import gym.wrappers
import lightning
import numpy as np
import torch
from gym import spaces
from gym.spaces import flatdim
from gym.wrappers import (
    clip_action,
    flatten_observation,
    normalize,
    transform_observation,
    transform_reward,
)
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo
from lightning import LightningModule
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from torch import Tensor, nn
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer

from project.algorithms.algorithm import Algorithm
from project.algorithms.ppo.dataloader_wrapper import PpoDataLoaderWrapper
from project.algorithms.ppo.utils import (
    PPOActorOutput,
    discount_cumsum,
)
from project.algorithms.rl_example.rl_datamodule import EpisodeBatch, RlDataModule
from project.algorithms.rl_example.utils import check_and_normalize_box_actions
from project.networks.fcnet import FcNet
from project.utils.types import PhaseStr, StepOutputDict

logger = get_logger(__name__)
eps = np.finfo(np.float32).eps.item()


def init_linear_layers(module: nn.Module, std=np.sqrt(2), bias_const=0.0):
    for layer in [module] + list(module.children()):
        if isinstance(layer, nn.Linear):
            init_linear_layer(layer, std, bias_const)


def init_linear_layer(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


# IDEA: Create a subclass of `Normal` that can handle having nested tensors as inputs.
# class Normal(torch.distributions.Normal):
#     def __init__(self, loc, scale, validate_args=None):
#         super().__init__(loc, scale, validate_args)


class PPO(Algorithm):
    """PPO Algorithm, based on the CleanRL implementation.

    See [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)


    NOTE: definitions:
    - "round": one cycle of gathering the data using the policy and then training for a
      few epochs on it.
    - "epoch": One pass through the data collected in this round. Each epoch in a round uses
      a different random ordering of episodes. (NOTE: This is different from the CleanRL
      implementation which shuffles steps within episodes).

    ```pseudocode
    for round in range(args.num_rounds):
        episodes = collect_steps_from_env(n_steps=args.num_steps)
        episode_indices = np.arange(len(episodes))

        for epoch_in_round in range(args.update_epochs):
            np.random.shuffle(episode_indices)

            for start_index, end_index in range(0, len(batch), args.minibatch_size):
                minibatch = episodes[episode_indices[start_index:end_index]]

                loss = training_loss(minibatch)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
    ```

    Here, we use an adapter around the train dataloader that accumulates episodes in bursts, and
    then yields minibatches from the accumulated episodes.

    This lines up with the CleanRL algo: the `training_step` becomes the inner for loop of the
    `update_epochs` updates.
    """

    @dataclass
    class HParams(Algorithm.HParams):
        """Hyperparameters for the PPO algorithm.

        These are taken directly from the CleanRL PPO implementation.
        """

        value_network: FcNet.HParams = field(
            default_factory=partial(
                FcNet.HParams, hidden_dims=[64, 64], activation="tanh", dropout_rate=0.0
            )
        )

        lam: float = 0.95
        """Lambda for GAE-Lambda.

        (Always between 0 and 1, close to 1.)
        """

        total_timesteps: int = 1000000
        """Total timesteps of the experiments."""
        num_envs: int = 1
        """The number of parallel game environments."""
        num_steps: int = 2048
        """The number of steps to run in each environment per policy rollout."""
        num_minibatches: int = 32
        """The number of mini-batches."""
        update_epochs: int = 10
        """The K epochs to update the policy."""

        learning_rate: float = 3e-4
        """The learning rate of the optimizer."""
        anneal_lr: bool = True
        """Toggle learning rate annealing for policy and value networks."""
        gamma: float = 0.99
        """The discount factor gamma."""
        gae_lambda: float = 0.95
        """The lambda for the general advantage estimation."""
        norm_adv: bool = True
        """Toggles advantages normalization."""
        clip_coef: float = 0.2
        """The surrogate clipping coefficient.

        Roughly: how far can the new policy go from the old policy while still profiting (improving
        the objective function)? The new policy can still go farther than the clip_ratio says, but
        it doesn't help on the objective anymore. (Usually small, 0.1 to 0.3.) Typically denoted
        by ɛ.
        """
        clip_vloss: bool = True
        """Toggles whether or not to use a clipped loss for the value function, as per the
        paper."""
        ent_coef: float = 0.0
        """Coefficient of the entropy."""
        vf_coef: float = 0.5
        """Coefficient of the value function."""
        max_grad_norm: float = 0.5
        """The maximum norm for the gradient clipping."""
        # target_kl: float | None = None
        # """The target KL divergence threshold."""

    def __init__(
        self,
        datamodule: RlDataModule,
        network: FcNet,
        hp: PPO.HParams | None = None,
    ):
        super().__init__(datamodule, network, hp=hp)
        self.hp: PPO.HParams
        self.network: FcNet
        self.datamodule: RlDataModule[PPOActorOutput] = self.datamodule.set_actor(
            actor=self.forward
        )

        # NOTE: assuming continuous actions for now.
        env = self.datamodule.env
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)
        # NOTE: We later add a wrapper that normalizes the action space to [-1, 1]
        self._action_space: spaces.Box = spaces.Box(-1, 1, shape=env.action_space.shape)
        action_dims = flatdim(env.action_space)

        assert isinstance(network, FcNet), "Assuming a fully connected network for now"
        if network.output_dims != action_dims:
            last_layer = network[-1]
            assert isinstance(last_layer, nn.Linear)
            self.network = copy.deepcopy(network)
            self.network[-1] = nn.Linear(
                last_layer.in_features, action_dims, bias=last_layer.bias is not None
            )
            self.network.output_dims = action_dims
            self.network.output_shape = (action_dims,)
            # TODO: Not ideal that the FcNet is created differently between Reinforce and PPO..
            logger.warning(
                RuntimeWarning(
                    f"Modifying the network's output shape from {network.output_shape} to "
                    f"{self.network.output_shape} since PPO doesn't use the network to predict "
                    f"the action standard deviation. "
                )
            )

        # Note: `actor_mean` is just an alias for `self.network`
        self.actor_mean = self.network
        self.actor_logstd = nn.Parameter(torch.zeros(action_dims))
        self.value_network = FcNet(
            input_shape=env.observation_space.shape,
            output_shape=(1,),
            hparams=self.hp.value_network,
        )

        # The FcNets of PPO have a custom initialization to match the CleanRL PPO.
        init_linear_layers(self.actor_mean)
        last_actor_mean_layer = self.actor_mean[-1][0]
        assert isinstance(last_actor_mean_layer, nn.Linear)
        init_linear_layer(last_actor_mean_layer, std=0.01)

        init_linear_layers(self.value_network)
        last_value_layer = self.value_network[-1][0]
        assert isinstance(last_value_layer, nn.Linear)
        init_linear_layer(last_value_layer, std=1.0)

        max_episode_length = self.datamodule.env.spec.max_episode_steps
        assert max_episode_length is not None, "TODO: assumed for now to simplify the code a bit."
        assert self.hp.num_envs == 1, "TODO: Only support running on a single env atm."

        # (Converting the loop lengths and batch sizes from CleanRL)
        steps_per_round = self.hp.num_envs * self.hp.num_steps

        self.num_rounds = math.ceil(self.hp.total_timesteps / steps_per_round)
        self.epochs_per_round = self.hp.update_epochs
        self.num_epochs = self.num_rounds * self.epochs_per_round

        self.episodes_per_round = math.ceil(steps_per_round / max_episode_length)
        # Note: This is the same since we shuffle and re-yield the gathered episodes multiple times
        episodes_per_epoch = self.episodes_per_round
        batches_per_epoch = self.hp.num_minibatches
        num_episodes_per_batch = math.ceil(episodes_per_epoch / batches_per_epoch)

        # Instruct the datamodule's train dataloader to collect this many episodes per __iter__.
        # This means that the "base" train dataloader will be exhausted after collecting this many
        # epsiodes, which also lines up with the number of episodes we want to collect per "round"
        # in our dataloader adapter below.
        self.datamodule.episodes_per_epoch = self.episodes_per_round
        # Wrap the training dataloader so it collects episodes in bursts and then in yield
        # minibatches.
        self.datamodule.train_dataloader_wrappers = [
            lambda dataloader: PpoDataLoaderWrapper(
                dataloader,
                min_steps_to_collect_per_round=steps_per_round,
                num_epochs_per_round=self.epochs_per_round,
                num_episodes_per_batch=num_episodes_per_batch,
            )
        ]

    @property
    def actor_mean(self) -> FcNet:
        return self.network

    @actor_mean.setter
    def actor_mean(self, value: FcNet) -> None:
        self.network = value

    def on_fit_start(self) -> None:
        logger.info("Starting training.")
        # We only add the gym wrappers to the datamodule once at the start of training.
        assert self.datamodule.train_dataset is None
        assert len(self.datamodule.train_wrappers) == 0
        self.datamodule.train_wrappers.extend(self.gym_wrappers_to_add(videos_subdir="train"))
        self.datamodule.valid_wrappers.extend(self.gym_wrappers_to_add(videos_subdir="valid"))
        self.datamodule.test_wrappers.extend(self.gym_wrappers_to_add(videos_subdir="test"))

    def gym_wrappers_to_add(self, videos_subdir: str) -> list[Callable[[gym.Env], gym.Env]]:
        clip_function = partial(np.clip, a_min=-10, a_max=10)
        video_folder = str(self.log_dir / "videos" / videos_subdir)
        return [
            # NOTE: The functools.partial is a pickelable equivalent to this:
            # lambda env: RecordVideo(env, video_folder=video_folder),
            partial(RecordVideo, video_folder=video_folder),
            RecordEpisodeStatistics,
            flatten_observation.FlattenObservation,
            clip_action.ClipAction,
            normalize.NormalizeObservation,
            partial(transform_observation.TransformObservation, f=clip_function),
            partial(normalize.NormalizeReward, gamma=self.hp.gamma),
            partial(transform_reward.TransformReward, f=clip_function),
            check_and_normalize_box_actions,
        ]

    def forward(
        self,
        observations: Tensor,
        action_space: spaces.Space[Tensor],
    ) -> tuple[Tensor, PPOActorOutput]:
        # NOTE: Would be nice to be able to do this:
        # assert observations.shape == self.network.input_space.shape
        # assert action_space.n == self.network.output_space.shape[0]

        # if observations.is_nested:
        #     observations, success = make_dense_if_possible(observations)
        #     if not success:
        #         raise NotImplementedError(
        #             "Can't pass nested tensor to torch.distributions.Normal yet."
        #             "Therefore we need to have the same shape for all the nested tensors."
        #         )
        assert isinstance(action_space, spaces.Box)
        assert (action_space.low == -1).all() and (action_space.high == 1).all()
        assert self.network.output_dims == action_space.shape[-1]

        action_mean = self.actor_mean(observations)
        action_logstd = self.actor_logstd.view_as(action_mean)
        values = self.value_network(observations)
        # TODO: Might need to adapt this below if we re-add support for nested tensors.
        action_std = torch.exp(action_logstd)
        action_distribution = Normal(action_mean, action_std)
        actions = action_distribution.sample()

        # TODO: How much stuff do we want to pre-compute here, versus doing it in the shared step?
        # Do we re-calculate these values for the behaviour policy in the shared step? If so,
        # then we might as well calculate them only once here.

        # NOTE: sum(-1) because each action dimension is treated as independent.
        action_log_probabilities = action_distribution.log_prob(actions).sum(-1)
        # action_distribution_entropy = action_distribution.entropy().sum(-1)

        return actions, PPOActorOutput(
            action_mean=action_mean,
            action_std=action_std,
            action_log_probability=action_log_probabilities,
            values=values,
        )

    def get_action_distribution(self, network_outputs: Tensor, action_space: spaces.Box) -> Normal:
        """Creates an action distribution based on the network outputs."""
        # NOTE: The environment has a wrapper applied to it that normalizes the continuous action
        # space to be in the [-1, 1] range, and the actions outside that range will be clipped by
        # that wrapper.

        # loc, scale = network_outputs.chunk(2, -1)
        # loc = torch.tanh(loc)
        # scale = torch.relu(scale) + 1e-5
        action_mean = network_outputs
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def configure_optimizers(
        self,
    ) -> dict:
        # TODO: Configure the learning rate annealing.
        LightningModule.configure_optimizers

        def get_lr_for_epoch(epoch: int) -> float:
            round = epoch // self.epochs_per_round
            frac = 1.0 - (round - 1.0) / (self.num_rounds)
            return frac * self.hp.learning_rate

        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hp.learning_rate,
            eps=1e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optim,
            lr_lambda=get_lr_for_epoch,
            # last_epoch=self.num_epochs,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": self.epochs_per_round,
            },
        }

    def training_step(self, batch: EpisodeBatch[PPOActorOutput]) -> StepOutputDict:
        return self.shared_step(batch, phase="train")

    def validation_step(
        self,
        batch: EpisodeBatch[PPOActorOutput],
        batch_index: int,
    ) -> StepOutputDict:
        return self.shared_step(batch, phase="val")

    @property
    def current_round(self) -> int:
        return self.current_epoch // self.epochs_per_round

    @property
    def current_epoch_in_round(self) -> int:
        return self.current_epoch % self.epochs_per_round

    def shared_step(self, batch: EpisodeBatch[PPOActorOutput], phase: PhaseStr) -> StepOutputDict:
        """A single learning step of PPO.

        A total of `self.hp.

        The data here is a minibatch of episodes, sampled from the current "round" of data.

        Parameters
        ----------
        batch: A batch of episodes (sampled from the "pool" of gathered episodes for this epoch)
        phase: The current phase (one of "train", "val" or "test").
        optimizer_idx: (unused atm): Index of the current optimizer.

        Returns
        -------
        StepOutputDict
            _description_
        """
        # steps_in_batch = sum(get_episode_lengths(batch))

        # NOTE: "b": "behaviour", "t": "target"
        # TODO: Add LR annealing https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L133-L135
        # TODO: Also anneal the clip range: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#LL137C9-L137C21
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        batch_size = rewards.size(0)
        assert batch_size > 0, batch
        # self.total_steps += total_steps_in_batch

        b_actor_outputs = batch["actor_outputs"]
        b_values = b_actor_outputs["values"].view_as(rewards)
        b_action_log_probs = b_actor_outputs["action_log_probability"]
        returns = discount_cumsum(rewards, discount=self.hp.gamma)

        # TODO: CleanRL's PPO uses the advantage value somehow when setting `returns`!
        # TODO: Need the bootstrap value for the last observation..

        t_values = self.value_network(observations)
        bootstrap_value = (
            t_values[:, -1]
            # if not t_values.is_nested
            # else torch.stack([values_i[-1] for values_i in t_values.unbind()])
        )
        b_advantages = self.compute_advantages(
            rewards=rewards, values=b_values, bootstrap_value=bootstrap_value
        )
        assert b_advantages.size(0) == batch_size

        observations_without_last = observations[..., :-1, :]  # FIXME
        t_action_logits = self.network(observations_without_last)
        t_action_dist = self.get_action_distribution(
            t_action_logits, action_space=self._action_space
        )
        t_action_log_probs = t_action_dist.log_prob(actions).sum(-1)
        # t_advantages = self.compute_advantages(rewards=rewards, values=t_values)

        t_entropy = t_action_dist.entropy().sum(1)

        logratio = t_action_log_probs - b_action_log_probs
        assert t_action_log_probs.shape == b_action_log_probs.shape

        ratio = logratio.exp()
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.hp.clip_coef).float().mean()
            self.log(f"{phase}/old_approx_kl", old_approx_kl, batch_size=batch_size)
            self.log(f"{phase}/approx_kl", approx_kl, batch_size=batch_size)
            self.log(f"{phase}/clip_fraction", clipfracs, batch_size=batch_size)

        if self.hp.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.hp.clip_coef, 1 + self.hp.clip_coef)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        # t_values = t_values.view(-1)
        # from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py#LL243C15-L243C15
        b_returns = b_advantages + b_values
        # TODO: Double-check this: `t_values` has a value for the final observation.
        t_values_without_last = t_values[:, :-1, 0]  # FIXME: Check the indexing
        if self.hp.clip_vloss:
            v_loss_unclipped = (t_values_without_last - b_returns) ** 2
            v_clipped = b_values + torch.clamp(
                t_values_without_last - b_values, -self.hp.clip_coef, self.hp.clip_coef
            )
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((t_values_without_last - b_returns) ** 2).mean()

        # Entropy loss
        entropy_loss = -1 * t_entropy.mean()

        entropy_loss_term = self.hp.ent_coef * entropy_loss
        value_loss_term = self.hp.vf_coef * v_loss

        self.log(f"{phase}/policy_loss", policy_loss, batch_size=batch_size)
        self.log(f"{phase}/entropy_loss", entropy_loss_term, batch_size=batch_size)
        self.log(f"{phase}/value_loss", value_loss_term, batch_size=batch_size)

        loss = policy_loss + self.hp.ent_coef * entropy_loss + self.hp.vf_coef * v_loss
        self.log(f"{phase}/loss", loss, batch_size=batch_size)

        # optimizer.zero_grad()
        # loss.backward()
        # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        # optimizer.step()

        # Log the episode statistics gathered by the RecordEpisodeStatistics gym wrapper.
        episode_stats = [
            episode_infos[-1]["episode"]
            for episode_infos in batch["infos"]
            if episode_infos and "episode" in episode_infos[-1]
        ]
        if episode_stats:
            episode_lengths = np.array([s["l"] for s in episode_stats])
            episode_total_rewards = np.array([s["r"] for s in episode_stats])
            # episode_time_since_start = np.array([s["t"] for s in episode_stats])  # unused atm.
            avg_episode_length = sum(episode_lengths) / batch_size
            avg_episode_reward = episode_total_rewards.mean(0)
            avg_episode_return = sum(returns.select(dim=1, index=0)) / batch_size
            # log_kwargs = dict(prog_bar=True, batch_size=batch_size)
            self.log(f"{phase}/avg_episode_length", avg_episode_length, batch_size=batch_size)
            self.log(f"{phase}/avg_episode_reward", avg_episode_reward, batch_size=batch_size)
            self.log(
                f"{phase}/avg_episode_return",
                avg_episode_return,
                batch_size=batch_size,
                prog_bar=True,
            )

        return {"loss": loss}

    def compute_advantages(
        self, rewards: Tensor, values: Tensor, bootstrap_value: Tensor
    ) -> Tensor:
        # values = values.reshape_as(rewards)
        assert rewards.is_nested == values.is_nested
        values = values.view_as(rewards)

        if not rewards.is_nested:
            values_tp1 = torch.cat([values, bootstrap_value], dim=1)
            deltas = rewards + self.hp.gamma * values_tp1[:, 1:] - values_tp1[:, :-1]
            return discount_cumsum(deltas, discount=self.hp.gamma * self.hp.lam, dim=1)

        values_tp1 = torch.nested.as_nested_tensor(
            [
                torch.cat([values_i, bootstrap_i], dim=0)
                for values_i, bootstrap_i in zip(values.unbind(), bootstrap_value.unbind())
            ]
        )
        deltas = torch.nested.as_nested_tensor(
            [
                rewards_i + self.hp.gamma * values_i[1:] - values_i[:-1]
                for rewards_i, values_i in zip(rewards.unbind(), values_tp1.unbind())
            ]
        )
        advantage = discount_cumsum(deltas, discount=self.hp.gamma * self.hp.lam, dim=1)
        return advantage

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        optimizer_idx: int,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        if optimizer_idx == 0:
            # Use the value passed to the `Trainer` constructor if not specified in the hparams of
            # this algo.
            self.clip_gradients(
                optimizer=optimizer,
                gradient_clip_val=self.hp.max_grad_norm or gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    @property
    def log_dir(self) -> Path:
        """Returns  the Trainer's log dir if we have a trainer.

        (NOTE: we should always have one, except maybe during some unit tests where the DataModule
        is used by itself.)
        """
        log_dir = Path("logs/default")
        if self.trainer is not None:
            log_dir = Path(self.trainer.log_dir or log_dir)
        return log_dir


def main():
    import rich.logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s ",
        handlers=[rich.logging.RichHandler()],
    )
    logging.getLogger("project").setLevel(logging.DEBUG)
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    datamodule = RlDataModule(
        env="Pendulum-v1", actor=None, episodes_per_epoch=10_000, batch_size=1
    )

    # TODO: Test out if we can make this stuff work with Brax envs:
    # from brax import envs
    # import brax.envs.wrappers
    # from brax.envs import create
    # from brax.envs import wrappers
    # from brax.io import metrics
    # from brax.training.agents.ppo import train as ppo
    # env = create("halfcheetah", batch_size=2, episode_length=200, backend="spring")
    # env = wrappers.VectorGymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    # env = wrappers.TorchWrapper(env, device=torch.device("cuda"))

    network = FcNet(
        input_shape=datamodule.env.observation_space.shape,
        output_shape=datamodule.env.action_space.shape,
        hparams=FcNet.HParams(
            hidden_dims=[64, 64],
            activation="Tanh",
            dropout_rate=0.0,
        ),
    )

    algorithm = PPO(datamodule=datamodule, network=network)

    datamodule.set_actor(algorithm)

    trainer = lightning.Trainer(
        max_epochs=algorithm.num_epochs,
        devices=1,
        accelerator="auto",
        default_root_dir="logs/debug",
        detect_anomaly=True,
        enable_checkpointing=False,
        check_val_every_n_epoch=10,
        callbacks=[RichProgressBar()],
    )
    trainer.fit(algorithm, datamodule=algorithm.datamodule)

    # Otherwise, could also do it manually, like so:

    # optim = algorithm.configure_optimizers()
    # for episode in algorithm.train_dataloader():
    #     optim.zero_grad()
    #     loss = algorithm.training_step(episode)
    #     loss.backward()
    #     optim.step()
    #     print(f"Loss: {loss}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import dataclasses
import time
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Literal, ParamSpec, Protocol, TypedDict, TypeVar

import chex
import flax.linen
import flax.struct
import gymnasium
import gymnasium.spaces
import gymnax
import gymnax.experimental.rollout
import jax
import lightning
import numpy as np
import torch
import torch_jax_interop
from flax.typing import FrozenVariableDict
from gymnasium import Space
from gymnasium.wrappers.record_video import RecordVideo
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from lightning import LightningModule, Trainer
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from project.datamodules.rl.datamodule import EnvDataLoader, RlDataModule
from project.datamodules.rl.envs import make_torch_vectorenv
from project.datamodules.rl.episode_dataset import EpisodeIterableDataset
from project.datamodules.rl.stacking_utils import NestedCategorical
from project.datamodules.rl.types import (
    Actor,
    Episode,
    EpisodeBatch,
    Transition,
    VectorEnv,
)
from project.datamodules.rl.wrappers.normalize_actions import (
    check_and_normalize_box_actions,
)
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorMultiDiscrete,
    TensorSpace,
)
from project.utils.device import default_device
from project.utils.typing_utils import NestedMapping
from project.utils.typing_utils.protocols import Module

logger = get_logger(__name__)
# torch.set_float32_matmul_precision("high")
eps = np.finfo(np.float32).eps.item()


class ReinforceActorOutput(TypedDict):
    """Additional outputs of the Actor (besides the action to take) for a single step in the env.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)

    In the case of Reinforce, we store the logits as well as the action log probabilities.
    """

    logits: Tensor
    """The network outputs at that step."""

    action_distribution: torch.distributions.Distribution
    """The distribution over the action space form which the action was sampled."""


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((*x.shape[:-1], -1))


class JaxFcNet(flax.linen.Module):
    num_classes: int = 10
    num_features: int = 256

    @flax.linen.compact
    def __call__(self, x: jax.Array, forward_rng: chex.PRNGKey | None = None):
        x = flatten(x)
        x = flax.linen.Dense(features=self.num_features)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


P = ParamSpec("P")
Out = TypeVar("Out", covariant=True)


class _Module(Protocol[P, Out]):
    if typing.TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> Out: ...

        init = flax.linen.Module.init
        apply = flax.linen.Module.apply


class Episodes(flax.struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array
    cum_ret: jax.Array


class RolloutWrapper(gymnax.experimental.rollout.RolloutWrapper):
    def __init__(
        self,
        env: Environment,
        env_params: gymnax.EnvParams,
        model_forward: Callable,
    ):
        # Define the RL environment & network forward function
        self.env = env
        self.env_params = env_params
        self.model_forward = model_forward

        self.num_env_steps = self.env_params.max_steps_in_episode


def get_episodes(
    env: gymnax.environments.environment.Environment[TEnvState, TEnvParams],
    env_params: TEnvParams,
    num_episodes: int,
    model: _Module[[jax.Array, chex.PRNGKey], jax.Array],
    model_params: FrozenVariableDict | dict[str, Any],
    rollout_rng_key: chex.PRNGKey,
) -> Episodes:
    rollout_wrapper = RolloutWrapper(
        env=env,
        env_params=env_params,
        model_forward=model.apply,
    )

    rollout_rng_keys = jax.random.split(rollout_rng_key, num_episodes)
    obs, action, reward, next_obs, done, cum_ret = rollout_wrapper.batch_rollout(
        rollout_rng_keys, model_params
    )
    return Episodes(
        obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, cum_ret=cum_ret
    )

    # return EpisodeBatch(
    #     observations=obs,
    #     actions=action,
    #     rewards=reward,
    #     final_observations=next_obs,
    # )


def main():
    env_id = "Pendulum-v1"

    from brax.envs import _envs as brax_envs
    from rejax.compat.brax2gymnax import create_brax

    if env_id in brax_envs:
        env, env_params = create_brax(
            env_id,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
            batch_size=None,
        )
    else:
        env, env_params = gymnax.make(env_id=env_id)

    algo = RlExample(env=env)
    trainer = lightning.Trainer(max_epochs=1)
    trainer.fit(algo)
    metrics = trainer.validate(algo)
    print(metrics)
    return


class RlExample(lightning.LightningModule):
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.env_params = env.default_params

        # https://github.com/RobertTLange/gymnax-blines/blob/main/utils/ppo.py
        sample_obs = env.observation_space(self.env_params).sample(jax.random.key(1))
        action_shape = env.action_space(self.env_params).shape

        net_rng = jax.random.key(123)
        self.network = JaxFcNet(num_classes=int(np.prod(action_shape)), num_features=256)
        self.network_params = self.network.init(net_rng, sample_obs, forward_rng=None)
        self.rollout_rng_key = jax.random.key(123)

        self.torch_network_params = torch.nn.ParameterList(
            jax.tree.leaves(
                jax.tree.map(torch_jax_interop.to_torch.jax_to_torch_tensor, self.network_params)
            )
        )

        self.automatic_optimization = False

    def train_dataloader(self) -> Any:
        for batch_idx in range(10):
            episode_key = jax.random.fold_in(self.rollout_rng_key, batch_idx)
            yield get_episodes(
                env=self.env,
                env_params=self.env_params,
                num_episodes=10,
                model=self.network,
                model_params=self.network_params,
                rollout_rng_key=episode_key,
            )

    def training_step(self, batch: Episodes, batch_idx: int):
        assert False, batch

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    main()
    exit()


class Reinforce(LightningModule):
    """Example of a Reinforcement Learning algorithm: Reinforce."""

    @dataclass
    class HParams:
        gamma: float = 0.99
        learning_rate: float = 1e-2

    def __init__(
        self,
        datamodule: RlDataModule | None,
        network: Module[[Tensor], Tensor],
        hp: Reinforce.HParams | None = None,
    ):
        """
        Parameters
        ----------

        - env: Gym environment to train on.
        - gamma: Discount rate.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        self.hp = hp or self.HParams()
        self._train_loader: EnvDataLoader[ReinforceActorOutput] | None = None
        self.save_hyperparameters(dataclasses.asdict(self.hp))

        self.automatic_optimization = False

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.hp.learning_rate)

    def forward(
        self,
        observations: Tensor,
        action_space: gymnasium.Space[Tensor],
    ) -> tuple[Tensor, ReinforceActorOutput]:
        """Forward pass: Given some observations, return an action some additional outputs.

        Parameters
        ----------
        observations: Either a single observation, or a batch of observations when training with \
            a vectorized environment.
        action_space: The space of possible actions. This is a version of gymnasium.Space where \
            sampling produces a Tensor on the same device as the environment.
        """
        network_outputs = self.network(observations.to(self.dtype))
        action_distribution = get_action_distribution(network_outputs, action_space)
        actor_outputs: ReinforceActorOutput = {
            "logits": network_outputs,
            "action_distribution": action_distribution,
        }
        actions = action_distribution.sample()
        return actions, actor_outputs

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        super().on_before_zero_grad(optimizer)
        # if self.datamodule is not None:
        #     logger.info("Updating the actor.")
        #     self.datamodule.on_actor_update()

    def training_step(self, batch: EpisodeBatch):
        step_outputs = self.shared_step(batch, phase="train")
        loss = step_outputs.pop("loss")
        optimizer = self.optimizers()
        assert isinstance(optimizer, Optimizer)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        logger.info(f"Updating the actor! {self.global_step=}")
        assert (
            self._train_dataset is not None
        ), "TODO: Using the .train_dataloader() method to create the dataloader instead of a datamodule"
        self._train_dataset.on_actor_update()
        # self._train_loader.on_actor_update()
        return {"loss": loss.detach(), **step_outputs}

    def transfer_batch_to_device(
        self, batch: Episode, device: torch.device, dataloader_idx: int
    ) -> Any:
        assert batch.observations.device == self.device
        return batch

    # NOTE: For some reason PL requires us to have a second positional argument for the batch_index
    # even if it isn't used, but the training step doesn't need it.
    def validation_step(self, batch: EpisodeBatch, batch_index: int):
        return self.shared_step(batch, phase="val")

    # def on_before_batch_transfer(self, batch: EpisodeBatch, dataloader_idx: int) -> EpisodeBatch:
    #     # IDEA: Use this PL hook to annotate the batch however you want.
    #     return batch

    def train_dataloader(self) -> Any:
        self._train_dataset: EpisodeIterableDataset[ReinforceActorOutput] = EpisodeIterableDataset(
            env=make_torch_vectorenv("CartPole-v1", num_envs=1, seed=123, device=self.device),
            actor=self,
        )
        # self._train_loader = EnvDataLoader(
        #     self._train_dataset,
        #     batch_size=1,
        # )
        return self._train_dataset

    @property
    def device(self) -> torch.device:
        """Small fixup for the `device` property in LightningModule, which is CPU by default."""
        if self._device.type == "cpu":
            self._device = next((p.device for p in self.parameters()), torch.device("cpu"))
        device = self._device
        # make this more explicit to always include the index
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", index=torch.cuda.current_device())
        return device

    def shared_step(self, batch: EpisodeBatch, phase: Literal["train", "val", "test"]):
        """Perform a single step of training or validation.

        The input is a batch of episodes, and the output is a dictionary with the loss and metrics.
        PyTorch-Lightning will then use the loss as the training signal, but we could also do the
        backward pass ourselves if we wanted to (as shown in the manual optimization example).
        """
        if isinstance(batch, Episode):
            batch = EpisodeBatch.from_episodes([batch])
        batch_size = batch.batch_size

        # Retrieve the outputs that we saved at each step:
        actor_outputs: ReinforceActorOutput = batch.actor_outputs  # type: ignore
        assert actor_outputs["logits"]._version == 0
        # Nested Tensor of shape [n_episodes, <episode_len>] where episode_len might vary between episodes.
        returns = batch.returns
        if returns is None:
            # When not using a PL trainer, for example in tests, the above hook isn't called.
            returns = get_returns(batch.rewards, gamma=self.hp.gamma)

        # NOTE: Equivalent to the following:
        if returns.is_nested:
            normalized_returns = torch.nested.as_nested_tensor(
                [(ret - ret.mean()) / (ret.std().clamp_min_(eps)) for ret in returns.unbind()]
            )
        else:
            normalized_returns = (returns - returns.mean(dim=1, keepdim=True)) / (
                returns.std(dim=1, keepdim=True).clamp_min_(eps)
            )

        # NOTE: In this particular case here, the actions are "live" tensors with grad_fns.
        # For Off-policy-style algorithms like DQN, this could be sampled from a replay buffer, and
        # so we could pass all the episodes through the network in a single forward pass (thanks to
        # the nested tensors).
        # Nested tensor of shape [n_envs, <episode_len>]
        action_distributions = actor_outputs["action_distribution"]
        action_log_probs = action_distributions.log_prob(batch.actions)
        if phase == "train":
            assert action_log_probs.requires_grad
        policy_loss_at_each_step = -action_log_probs * normalized_returns

        if policy_loss_at_each_step.is_nested:
            # Pad with zeros in case of different episode lengths.
            policy_loss_at_each_step = policy_loss_at_each_step.to_padded_tensor(0.0)

        # Sum within each episode, then average across episodes
        policy_loss = policy_loss_at_each_step.sum(dim=1).mean(dim=0)
        self.log(f"{phase}/loss", policy_loss, prog_bar=True, batch_size=batch_size)

        # Log some useful information.
        max_episode_length = max(batch.episode_lengths)
        avg_episode_length = sum(batch.episode_lengths) / batch_size
        _rewards = batch.rewards
        if _rewards.is_nested:
            _rewards = _rewards.to_padded_tensor(0.0)
        avg_episode_reward = _rewards.sum(1).mean(0)

        if not returns.is_nested:
            avg_episode_return = returns[:, 0].mean()
        else:
            avg_episode_return = torch.stack([ret[0] for ret in returns.unbind()]).mean()
        logs = {
            "max_episode_length": max_episode_length,
            "avg_episode_length": avg_episode_length,
            "avg_episode_reward": avg_episode_reward,
            "avg_episode_return": avg_episode_return,
        }
        for k, v in logs.items():
            self.log(
                f"{phase}/{k}",
                torch.as_tensor(v, dtype=torch.float32),
                prog_bar=True,
                batch_size=batch_size,
            )

        return {"loss": policy_loss, "log": logs}

    def on_fit_start(self) -> None:
        logger.info("Starting training.")
        if not self.datamodule:
            return
        assert isinstance(self.datamodule, RlDataModule) or hasattr(self.datamodule, "set_actor")
        # Set the actor on the datamodule so our `forward` method is used to select actions at each
        # step.
        self.datamodule.set_actor(self)

        # We only add the gym wrappers to the datamodule once.
        assert self.datamodule.train_dataset is None
        assert len(self.datamodule.train_wrappers) == 0
        self.datamodule.train_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="train"))
        self.datamodule.valid_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="valid"))
        self.datamodule.test_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="test"))

    def gym_wrappers_to_add(
        self, videos_subdir: str
    ) -> list[Callable[[gymnasium.Env], gymnasium.Env]]:
        video_folder = str(self.log_dir / "videos" / videos_subdir)
        logger.info(f"Saving videos in {video_folder}.")
        wrappers = [
            check_and_normalize_box_actions,
            # NOTE: The functools.partial below is Equivalent to the following:
            # lambda env: RecordVideo(env, video_folder=video_folder),
        ]
        if not self.datamodule:
            return []
        if self.datamodule.env.unwrapped.render_mode is not None:
            wrappers.append(lambda env: RecordVideo(env, video_folder=video_folder))
        return wrappers

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


def get_action_distribution(
    network_outputs: Tensor, action_space: Space[Tensor]
) -> torch.distributions.Distribution:
    """Creates an action distribution based on the network outputs."""
    # TODO: Once we can work with batched environments, should `action_space` here always be
    # the single action space?
    assert isinstance(action_space, TensorSpace), action_space

    if isinstance(action_space, TensorDiscrete | TensorMultiDiscrete):
        if network_outputs.is_nested:
            return NestedCategorical(logits=network_outputs)
        return Categorical(logits=network_outputs)

    # NOTE: The environment has a wrapper applied to it that normalizes the continuous action
    # space to be in the [-1, 1] range, and the actions outside that range will be clipped by
    # that wrapper.
    assert isinstance(action_space, TensorBox)
    assert (action_space.low == -1).all() and (action_space.high == 1).all(), action_space
    d = action_space.shape[-1]
    assert network_outputs.size(-1) == d * 2

    # todo: make sure that this works with nested tensors.
    loc, scale = network_outputs.chunk(2, -1)
    loc = torch.tanh(loc)
    scale = torch.relu(scale) + 1e-5

    return Normal(loc=loc, scale=scale)


def get_returns(rewards: Tensor, gamma: float, bootstrap_value: Tensor | float = 0.0) -> Tensor:
    """Returns a batch of discounted returns for each step of each episode.

    Parameters
    ----------
    rewards_batch: A (possibly nested) tensor of shape [b, `ep_len`] where `ep_len` may vary.
    gamma: The discount factor.

    Returns
    -------
    A (possibly nested) tensor of shape [b, `ep_len`] where `ep_len` may vary.
    """
    # todo: Check if this also works if the rewards batch is a regular tensor (with all the
    # episodes having the same length).
    # _batch_size, ep_len = rewards.shape
    if rewards.ndim not in (1, 2):
        raise ValueError(f"Expected either 1d or 2d rewards tensor, not {rewards.ndim}d tensor!")

    if not rewards.is_nested:
        was_1d = rewards.ndim == 1
        if was_1d:
            rewards = rewards.unsqueeze(0)
        returns = torch.zeros_like(rewards)
        discounted_sum_of_future_rewards = torch.ones_like(rewards[:, 0]) * bootstrap_value
        ep_len = rewards.size(-1)
        for step in reversed(range(ep_len)):
            discounted_sum_of_future_rewards = (
                rewards[..., step] + gamma * discounted_sum_of_future_rewards
            )
            returns[..., step] = discounted_sum_of_future_rewards
        if was_1d:
            returns = returns.squeeze(0)
        return returns

    return torch.nested.as_nested_tensor(
        [
            get_returns(episode_rewards, gamma, bootstrap_value)
            for episode_rewards in rewards.unbind()
        ]
    )


ActorOutput = TypeVar("ActorOutput", bound=NestedMapping[str, Tensor])


def collect_transitions(
    env_id: str,
    actor: Actor[Tensor, Tensor, ActorOutput],
    num_transitions: int,
    num_parallel_envs: int = 128,
    seed: int = 123,
    device: torch.device = default_device(),
) -> Sequence[Transition[ActorOutput]]:
    env = make_torch_vectorenv(env_id, num_envs=num_parallel_envs, seed=seed, device=device)
    dataset = EpisodeIterableDataset(env, actor=actor, steps_per_epoch=num_transitions)
    transitions: list[Transition[ActorOutput]] = []
    for episode in iter(dataset):
        transitions.extend(episode.as_transitions())
        if len(transitions) >= num_transitions:
            break
    return transitions


def collect_episodes(
    env: gymnasium.Env[Tensor, Tensor] | VectorEnv[Tensor, Tensor],
    *,
    actor: Actor[Tensor, Tensor, ActorOutput],
    min_episodes: int | None = None,
    min_num_transitions: int | None = None,
) -> EpisodeBatch[ActorOutput]:
    assert min_episodes or min_num_transitions
    dataset = EpisodeIterableDataset(
        env,
        actor=actor,
        steps_per_epoch=min_num_transitions,
        episodes_per_epoch=min_episodes,
    )
    episodes: list[Episode[ActorOutput]] = []
    for episode in iter(dataset):
        episodes.append(episode)
    return EpisodeBatch[ActorOutput].from_episodes(episodes)


class MeasureThroughputCallback(lightning.Callback):
    def __init__(self):
        super().__init__()
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()
        self._updates = 0

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()
        self._updates = 0

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer, opt_idx: int
    ) -> None:
        self._updates += 1

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: EpisodeBatch[dict],
        batch_idx: int,
    ) -> None:
        episodes = batch

        num_episodes = episodes.batch_size
        num_transitions = sum(episodes.episode_lengths)

        self.total_transitions += num_transitions
        self.total_episodes += num_episodes

        sps = self.total_transitions / (time.perf_counter() - self._start)
        updates_per_second = (self._updates) / (time.perf_counter() - self._start)
        episodes_per_second = self.total_episodes / (time.perf_counter() - self._start)

        pl_module.log_dict(
            {
                "sps": torch.as_tensor(sps, dtype=torch.float32),
                "ups": torch.as_tensor(updates_per_second, dtype=torch.float32),
                "eps": torch.as_tensor(episodes_per_second, dtype=torch.float32),
            },
            prog_bar=True,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        steps_per_second = self.total_transitions / (time.perf_counter() - self._start)
        updates_per_second = (self._updates) / (time.perf_counter() - self._start)
        episodes_per_second = self.total_episodes / (time.perf_counter() - self._start)
        logger.info(
            f"Total transitions: {self.total_transitions}, total episodes: {self.total_episodes}"
        )
        logger.info(f"Steps per second: {steps_per_second}")
        logger.info(f"Episodes per second: {episodes_per_second}")
        logger.info(f"Updates per second: {updates_per_second}")

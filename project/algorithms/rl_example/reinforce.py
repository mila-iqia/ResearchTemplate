from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, TypedDict

import gymnasium
import gymnasium.spaces
import lightning
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers.record_video import RecordVideo
from torch import Tensor
from torch.distributions import Categorical, Normal

from project.algorithms.bases.algorithm import Algorithm
from project.datamodules.rl.rl_datamodule import RlDataModule
from project.datamodules.rl.rl_types import EpisodeBatch
from project.datamodules.rl.wrappers.normalize_actions import check_and_normalize_box_actions
from project.datamodules.rl.wrappers.record_episode_statistics import RecordEpisodeStatistics
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorMultiDiscrete,
    TensorSpace,
)
from project.networks.fcnet import FcNet
from project.utils.types import PhaseStr, StepOutputDict
from project.utils.types.protocols import Module

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

    action_log_probability: Tensor
    """The log-probability of the selected action at that step."""


class Reinforce[ModuleType: Module[[Tensor], Tensor]](
    Algorithm[EpisodeBatch, StepOutputDict, ModuleType]
):
    """Example of a Reinforcement Learning algorithm: Reinforce.

    IDEA: Make this algorithm applicable in Supervised Learning by wrapping the
    dataset into a gym env, just to prove a point?
    """

    @dataclass
    class HParams(Algorithm.HParams):
        gamma: float = 0.99
        learning_rate: float = 1e-2

    def __init__(
        self,
        datamodule: RlDataModule,
        network: ModuleType,
        hp: Reinforce.HParams | None = None,
    ):
        """
        Parameters
        ----------

        - env: Gym environment to train on.
        - gamma: Discount rate.
        """
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.network: ModuleType
        self.hp: Reinforce.HParams
        self.datamodule: RlDataModule

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate)

    def forward(
        self,
        observations: Tensor,
        action_space: spaces.Discrete | spaces.Box,
    ) -> tuple[Tensor, ReinforceActorOutput]:
        # NOTE: Would be nice to be able to do this:
        # assert observations.shape == self.network.input_space.shape

        network_outputs = self.network(observations.to(torch.float32))

        if not observations.is_nested:
            # Either a single observation or a batch of observations.
            action_distribution = self.get_action_distribution(network_outputs, action_space)
            actions = action_distribution.sample()
            # (normalized) log probability of the selected actions (treated as independent).
            action_log_probabilities = action_distribution.log_prob(actions).sum(-1)
        else:
            # NOTE: This isn't used here, but could be very useful for off-policy algorithms like
            # for instance DQN:

            # Getting a batch of sequence of observations (a list[list[Observation]], probably a
            # list of episodes where episodes have different lengths) and we're returning the
            # actions that would have been taken for each observation.
            distributions = [
                self.get_action_distribution(outputs_i, action_space)
                for outputs_i in network_outputs.unbind()
            ]
            actions = torch.nested.as_nested_tensor([dist.sample() for dist in distributions])
            action_log_probabilities = torch.nested.as_nested_tensor(
                [dist.log_prob(a).sum(0) for dist, a in zip(distributions, actions.unbind())]
            )
        actor_outputs_to_save_in_episode: ReinforceActorOutput = {
            "logits": network_outputs,
            "action_log_probability": action_log_probabilities,
        }
        return actions, actor_outputs_to_save_in_episode

    def training_step(self, batch: EpisodeBatch) -> StepOutputDict:
        return self.shared_step(batch, phase="train")

    # NOTE: For some reason PL requires us to have a second positional argument for the batch_index
    # even if it isn't used, but the training step doesn't need it.
    def validation_step(self, batch: EpisodeBatch, batch_index: int) -> StepOutputDict:
        return self.shared_step(batch, phase="val")

    def shared_step(self, batch: EpisodeBatch, phase: PhaseStr) -> StepOutputDict:
        """Perform a single step of training or validation.

        The input is a batch of episodes, and the output is a dictionary with the loss and metrics.
        PyTorch-Lightning will then use the loss as the training signal, but we could also do the
        backward pass ourselves if we wanted to (as shown in the ManualGradientsExample).
        """
        rewards = batch["rewards"]
        # Retrieve the outputs that we saved at each step:
        actor_outputs = batch["actor_outputs"]
        batch_size = rewards.size(0)

        # Nested Tensor of shape [n_envs, <episode_len>] where episode_len varies between tensors.
        returns = discounted_returns(rewards, gamma=self.hp.gamma)

        # NOTE: Equivalent to the following:
        if returns.is_nested:
            normalized_returns = torch.nested.as_nested_tensor(
                [(ret - ret.mean()) / (ret.std() + eps) for ret in returns.unbind()]
            )
        else:
            normalized_returns = (returns - returns.mean(dim=1)) / (returns.std(dim=1) + eps)

        # NOTE: In this particular case here, the actions are "live" tensors with grad_fns.
        # For Off-policy-style algorithms like DQN, this could be sampled from a replay buffer, and
        # so we could pass all the episodes through the network in a single forward pass (thanks to
        # the nested tensors).

        # Nested tensor of shape [n_envs, <episode_len>]
        action_log_probs = actor_outputs["action_log_probability"].reshape_as(normalized_returns)
        policy_loss_per_step = -action_log_probs * normalized_returns

        # Sum across episode steps
        policy_loss_per_episode = policy_loss_per_step.to_padded_tensor(0.0).sum(dim=1)
        # Average across episodes
        policy_loss = policy_loss_per_episode.mean(dim=0)
        self.log(f"{phase}/loss", policy_loss, prog_bar=True)

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
            self.log(
                f"{phase}/avg_episode_length",
                avg_episode_length,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                f"{phase}/avg_episode_reward",
                avg_episode_reward,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                f"{phase}/avg_episode_return",
                avg_episode_return,
                prog_bar=True,
                batch_size=batch_size,
            )

        return {"loss": policy_loss}

    def on_fit_start(self) -> None:
        logger.info("Starting training.")
        assert isinstance(self.datamodule, RlDataModule) or hasattr(self.datamodule, "set_actor")
        # Set the actor on the datamodule so our `forward` method is used to select actions at each
        # step.
        self.datamodule.set_actor(self)

        # We only add the gym wrappers to the datamodule once.
        assert self.datamodule.train_dataset is None
        assert (
            len(self.datamodule.train_wrappers) == 0
            or self.datamodule.train_wrappers[-1] is not RecordEpisodeStatistics
        )
        self.datamodule.train_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="train"))
        self.datamodule.valid_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="valid"))
        self.datamodule.test_wrappers += tuple(self.gym_wrappers_to_add(videos_subdir="test"))

    def gym_wrappers_to_add(
        self, videos_subdir: str
    ) -> list[Callable[[gymnasium.Env], gymnasium.Env]]:
        video_folder = str(self.log_dir / "videos" / videos_subdir)
        wrappers = [
            check_and_normalize_box_actions,
            # NOTE: The functools.partial below is Equivalent to the following:
            # lambda env: RecordVideo(env, video_folder=str(log_dir / "videos/train")),
            RecordEpisodeStatistics,
        ]
        if self.datamodule.env.render_mode is not None:
            wrappers.append(functools.partial(RecordVideo, video_folder=video_folder))
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
        self,
        network_outputs: Tensor,
        action_space: TensorDiscrete | TensorBox | TensorMultiDiscrete,
    ) -> Categorical | Normal:
        """Creates an action distribution based on the network outputs."""
        # TODO: Once we can work with batched environments, should `action_space` here always be
        # the single action space?
        assert isinstance(action_space, TensorSpace), action_space

        if isinstance(action_space, TensorDiscrete | TensorMultiDiscrete):
            return Categorical(logits=network_outputs)

        # NOTE: The environment has a wrapper applied to it that normalizes the continuous action
        # space to be in the [-1, 1] range, and the actions outside that range will be clipped by
        # that wrapper.
        assert isinstance(action_space, TensorBox)
        assert (action_space.low == -1).all() and (action_space.high == 1).all(), action_space
        d = action_space.shape[-1]
        assert network_outputs.size(-1) == d * 2

        if network_outputs.is_nested:
            if not all(
                out_i.shape == network_outputs[0].shape for out_i in network_outputs.unbind()
            ):
                raise NotImplementedError(
                    "Can't pass a nested tensor to torch.distributions.Normal yet. "
                    "Therefore we need to have the same shape for all the nested tensors."
                )
            network_outputs = torch.stack(network_outputs.unbind())

        loc, scale = network_outputs.chunk(2, -1)
        loc = torch.tanh(loc)
        scale = torch.relu(scale) + 1e-5

        return Normal(loc=loc, scale=scale)


def discounted_returns(rewards_batch: Tensor, gamma: float) -> Tensor:
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

    # NOTE: `rewards` has shape [batch_size, <ep_length>] atm.
    assert rewards_batch.is_nested
    returns_batch: list[Tensor] = []

    for rewards in rewards_batch.unbind():
        returns = torch.zeros_like(rewards)

        discounted_future_rewards = torch.zeros_like(rewards[0])

        ep_len = rewards.size(0)

        for step in reversed(range(ep_len)):
            reward_at_that_step = rewards[step]
            discounted_future_rewards = reward_at_that_step + gamma * discounted_future_rewards
            returns[step] = discounted_future_rewards

        returns_batch.append(returns)

    return torch.nested.as_nested_tensor(returns_batch)


# def _discounted_returns_list(rewards: list[float], gamma: float) -> list[float]:
#     sum_of_discounted_future_rewards = 0
#     returns_list: deque[float] = deque()
#     for reward in reversed(rewards):
#         sum_of_discounted_future_rewards = reward + gamma * sum_of_discounted_future_rewards
#         returns_list.appendleft(sum_of_discounted_future_rewards)  # type: ignore
#     return list(returns_list)


def main():
    datamodule = RlDataModule(
        env="CartPole-v1", num_parallel_envs=128, actor=None, episodes_per_epoch=100, batch_size=10
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
        # todo: change to `input_dims` and pass flatdim(observation_space) instead.
        output_dims=gymnasium.spaces.flatdim(datamodule.env.action_space),
    )

    algorithm = Reinforce(datamodule=datamodule, network=network)

    datamodule.set_actor(algorithm)

    trainer = lightning.Trainer(max_epochs=2, devices=1, accelerator="auto")
    # todo: fine for now, but perhaps the SL->RL wrapper for Reinforce will change that.
    assert algorithm.datamodule is datamodule
    trainer.fit(algorithm, datamodule=datamodule)

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

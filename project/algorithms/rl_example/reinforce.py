from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, TypedDict

import gymnasium
import gymnasium.spaces
import numpy as np
import torch
from gymnasium import Space
from gymnasium.wrappers.record_video import RecordVideo
from lightning import LightningModule, Trainer
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.optim.optimizer import Optimizer

from project.algorithms.bases.algorithm import Algorithm
from project.algorithms.callbacks.callback import Callback
from project.datamodules.rl import episode_dataset
from project.datamodules.rl.datamodule import RlDataModule
from project.datamodules.rl.envs import make_torch_vectorenv
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
from project.utils.types import NestedMapping, PhaseStr, StepOutputDict
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

    action_distribution: torch.distributions.Distribution
    """The distribution over the action space form which the action was sampled."""

    # action_log_probability: Tensor
    # """The log-probability of the selected action at that step."""


class Reinforce(Algorithm[EpisodeBatch, StepOutputDict, Module[[Tensor], Tensor]]):
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
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.hp: Reinforce.HParams
        self.datamodule: RlDataModule | None

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate)

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
        actions = action_distribution.sample()
        actor_outputs: ReinforceActorOutput = {
            "logits": network_outputs,
            "action_distribution": action_distribution,
        }
        return actions, actor_outputs

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        super().on_before_zero_grad(optimizer)
        if self.datamodule is not None:
            logger.info("Updating the actor.")
            self.datamodule.on_actor_update()

    def training_step(self, batch: EpisodeBatch) -> StepOutputDict:
        return self.shared_step(batch, phase="train")

    # NOTE: For some reason PL requires us to have a second positional argument for the batch_index
    # even if it isn't used, but the training step doesn't need it.
    def validation_step(self, batch: EpisodeBatch, batch_index: int) -> StepOutputDict:
        return self.shared_step(batch, phase="val")

    # def on_before_batch_transfer(self, batch: EpisodeBatch, dataloader_idx: int) -> EpisodeBatch:
    #     # IDEA: Use this PL hook to annotate the batch however you want.
    #     return batch

    def shared_step(self, batch: EpisodeBatch, phase: PhaseStr) -> StepOutputDict:
        """Perform a single step of training or validation.

        The input is a batch of episodes, and the output is a dictionary with the loss and metrics.
        PyTorch-Lightning will then use the loss as the training signal, but we could also do the
        backward pass ourselves if we wanted to (as shown in the manual optimization example).
        """
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


def collect_transitions[ActorOutput: NestedMapping[str, Tensor]](
    env_id: str,
    actor: Actor[Tensor, Tensor, ActorOutput],
    num_transitions: int,
    num_parallel_envs: int = 128,
    seed: int = 123,
    device: torch.device = default_device(),
) -> Sequence[Transition[ActorOutput]]:
    env = make_torch_vectorenv(env_id, num_envs=num_parallel_envs, seed=seed, device=device)
    dataset = episode_dataset.EpisodeIterableDataset(
        env, actor=actor, steps_per_epoch=num_transitions
    )
    transitions: list[Transition[ActorOutput]] = []
    for episode in iter(dataset):
        transitions.extend(episode.as_transitions())
        if len(transitions) >= num_transitions:
            break
    return transitions


def collect_episodes[ActorOutput: NestedMapping[str, Tensor]](
    env: gymnasium.Env[Tensor, Tensor] | VectorEnv[Tensor, Tensor],
    *,
    actor: Actor[Tensor, Tensor, ActorOutput],
    min_episodes: int | None = None,
    min_num_transitions: int | None = None,
) -> EpisodeBatch[ActorOutput]:
    assert min_episodes or min_num_transitions
    dataset = episode_dataset.EpisodeIterableDataset(
        env,
        actor=actor,
        steps_per_epoch=min_num_transitions,
        episodes_per_epoch=min_episodes,
    )
    episodes: list[Episode[ActorOutput]] = []
    for episode in iter(dataset):
        episodes.append(episode)
    return EpisodeBatch[ActorOutput].from_episodes(episodes)


class MeasureThroughputCallback(Callback[EpisodeBatch, StepOutputDict]):
    def __init__(self):
        super().__init__()
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()
        self._updates = 0

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[EpisodeBatch[dict]],
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
        pl_module: Algorithm[EpisodeBatch[dict]],
        outputs: StepOutputDict,
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

    def on_fit_end(self, trainer: Trainer, pl_module: Algorithm[EpisodeBatch[dict]]) -> None:
        steps_per_second = self.total_transitions / (time.perf_counter() - self._start)
        updates_per_second = (self._updates) / (time.perf_counter() - self._start)
        episodes_per_second = self.total_episodes / (time.perf_counter() - self._start)
        logger.info(
            f"Total transitions: {self.total_transitions}, total episodes: {self.total_episodes}"
        )
        logger.info(f"Steps per second: {steps_per_second}")
        logger.info(f"Episodes per second: {episodes_per_second}")
        logger.info(f"Updates per second: {updates_per_second}")

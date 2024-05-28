from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, NotRequired, TypedDict

import gym
import gym.spaces
import gymnasium.spaces
import jax
import numpy as np
import torch
from gymnasium import Space
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import TypeVar

from project.utils.types import NestedMapping, is_list_of
from project.utils.utils import get_shape_ish

type _Env[ObsType, ActType] = gym.Env[ObsType, ActType] | gymnasium.Env[ObsType, ActType]
type _Space[T_cov] = gym.Space[T_cov] | gymnasium.Space[T_cov]

BoxSpace = gym.spaces.Box | gymnasium.spaces.Box
"""A Box space, either from Gym or Gymnasium."""

DiscreteSpace = gym.spaces.Discrete | gymnasium.spaces.Discrete
"""A Discrete space, from either Gym or Gymnasium."""


TensorType = TypeVar("TensorType", bound=Tensor, default=Tensor)
ObservationT = TypeVar("ObservationT", default=np.ndarray)
ActionT = TypeVar("ActionT", default=int)
# TypeVars for the observations and action types for a gym environment.

ActorOutput = TypeVar(
    "ActorOutput", bound=NestedMapping[str, Tensor], default=dict, covariant=True
)

Actor = Callable[[ObservationT, Space[ActionT]], tuple[ActionT, ActorOutput]]
"""An "Actor" is given an observation and action space, and asked to produce the next action.

It can also output other stuff that will be used to train the model later.
"""

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


### Typing fixes for gymnasium.


# VectorEnv doesn't have type hints in current gymnasium.
class VectorEnv[ObsType, ActType](gymnasium.vector.VectorEnv, gymnasium.Env[ObsType, ActType]):
    observation_space: Space[ObsType]
    action_space: Space[ActType]

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        return super().step(actions)

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict]:
        return super().reset(seed=seed, options=options)

    # could also perhaps add this render method?
    # def render[RenderFrame](self) -> RenderFrame | list[RenderFrame] | None:
    #     return self.call("render")


# Optional types for the last two typevars in Wrapper and VectorEnvWrapper.
WrappedObsType = TypeVar("WrappedObsType", default=Any)
WrappedActType = TypeVar("WrappedActType", default=Any)


# gymnasium.vector.VectorEnvWrapper doesn't have type hints in current gymnasium.
class VectorEnvWrapper(
    gymnasium.vector.VectorEnvWrapper,
    VectorEnv[WrapperObsType, WrapperActType],
    Generic[WrapperObsType, WrapperActType, WrappedObsType, WrappedActType],
):
    def __init__(self, env: VectorEnv[WrappedObsType, WrappedActType]):
        super().__init__(env)
        self.observation_space: Space[WrapperObsType]
        self.action_space: Space[WrapperActType]
        self.num_envs = env.num_envs

    # implicitly forward all other methods and attributes to self.env
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        # Remove this annoying warning.
        # logger.warn(
        #     f"env.{name} to get variables from other wrappers is deprecated and will be removed in v1.0, "
        #     f"to get this variable you can do `env.unwrapped.{name}` for environment variables."
        # )
        return getattr(self.env, name)


def random_actor(observation: Any, action_space: Space[ActionT]) -> tuple[ActionT, dict]:
    """Actor that takes random actions."""
    return action_space.sample(), {}


V = TypeVar("V", default=Any)


class MappingMixin(Mapping[str, V]):
    """Makes a dataclass usable like a Mapping."""

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __getitem__(self, key: str) -> V:
        return getattr(self, key)

    def keys(self):
        assert dataclasses.is_dataclass(self)
        return tuple(f.name for f in dataclasses.fields(self))

    def values(self) -> tuple[V, ...]:
        return tuple(v for _, v in self.items())

    def items(self) -> Sequence[tuple[str, V]]:
        assert dataclasses.is_dataclass(self)
        return {k: getattr(self, k) for k in self.keys()}.items()  # type: ignore


# todo: make this generic w.r.t. obs / act types once we add envs with more complicated observations.
@dataclass(frozen=True)
class Episode(MappingMixin, Generic[ActorOutput]):
    """An Episode where the contents have been stacked into tensors instead of kept in lists."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    # todo: Gymnasium wrappers assume that rewards is a numpy array for now. Might be easier to
    # use jax Arrays instead of torch Tensors here.

    infos: list[dict]
    """Info dicts at each step.

    Not stacked, since there might be different keys at every step.
    """

    # Should we even allow these to be on the GPU (jax or Torch tensors?) Or should they be forced
    # to be plain bools?
    truncated: bool
    terminated: bool

    actor_outputs: ActorOutput

    final_observation: Tensor | None = None
    final_info: dict | None = None

    environment_index: int | None = None
    """The environment index (when in a vectorenv) that this episode was generated from."""

    @property
    def length(self) -> int:
        size = self.rewards.shape[0]
        assert self.observations.size(0) == self.actions.size(0) == len(self.infos) == size
        return size

    def as_transitions(self):
        """Convert the episode into a sequence of `Transition`s."""
        from project.datamodules.rl.episode_dataset import sliced_dict

        sliced_actor_outputs = list(sliced_dict(self.actor_outputs, n_slices=self.length))
        return tuple(
            MiddleTransition(
                observation=self.observations[i],
                info=self.infos[i],
                action=self.actions[i],
                actor_output=sliced_actor_outputs[i],
                reward=self.rewards[i],
                next_observation=self.observations[i + 1],
                next_info=self.infos[i + 1],
            )
            for i in range(self.length - 1)
        ) + (
            FinalTransition(
                observation=self.observations[-1],
                info=self.infos[-1],
                action=self.actions[-1],
                actor_output=sliced_actor_outputs[-1],
                reward=self.rewards[-1],
                terminated=self.terminated,
                truncated=self.truncated,
                final_observation=self.final_observation,
                final_info=self.final_info,
            ),
        )

    def full_transitions(self) -> list[MiddleTransition[ActorOutput]]:
        """Convert the episode into a sequence of `Transition`s where every transition has."""
        *full_transitions, last_full_transition, _final_transition = self.as_transitions()
        return full_transitions + [dataclasses.replace(last_full_transition, is_terminal=True)]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transition[ActorOutput]:
    observation: Tensor
    info: dict
    action: Tensor
    actor_output: ActorOutput
    reward: Tensor | jax.Array
    is_terminal: bool


@dataclasses.dataclass(frozen=True, kw_only=True)
class MiddleTransition(Transition[ActorOutput]):
    next_observation: Tensor
    next_info: dict
    is_terminal: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class FinalTransition(Transition[ActorOutput]):
    terminated: bool | torch.BoolTensor | jax.Array
    truncated: bool | torch.BoolTensor | jax.Array
    final_observation: Tensor | None = None
    final_info: dict | None = None
    is_terminal: bool = True


@dataclass(frozen=True)
class EpisodeBatch(MappingMixin, Generic[ActorOutput]):
    """Object that contains a batch of episodes, stored as nested tensors.

    A [nested tensor](https://pytorch.org/docs/stable/nested.html#module-torch.nested) is basically
    a list of tensors, where the stored tensors may have different lengths.

    In our case, this is useful since episodes might have different number of steps, and it is more
    efficient to perform a single forward pass with a nested tensor than it would be to do a python
    for loop with multiple forward passes for each episode.

    See https://pytorch.org/docs/stable/nested.html#module-torch.nested for more info.
    """

    observations: Tensor  # [B, <episode_lengths>, *observation_shape]
    actions: Tensor  # [B, <episode_lengths>, *action_shape]
    rewards: Tensor  # [B, <episode_lengths>]

    infos: list[list[dict]]  # [B, <episode_lengths>]
    """List of the `info` dictionaries in each episode."""

    truncated: Tensor  # [B]
    """Whether each episodes was truncated (for example if a maximum number of steps was
    reached)."""
    terminated: Tensor  # [B]
    """Whether each episodes ended "normally"."""

    actor_outputs: ActorOutput  # Each value will have shape [B, <episode_lengths>]
    """Additional outputs of the Actor (besides the action to take) in each episode.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """

    final_infos: list[dict] | None
    """Info dicts at the final step of each episode, or `None` if that isn't saved in the env."""

    final_observations: Tensor | None
    """Stacked tensor with the final observation of each episode, or None if that isn't saved."""

    def shapes(self) -> dict[str, tuple[int, int | Literal["?"], *tuple[int, ...]]]:
        return {k: get_shape_ish(v) for k, v in self.items() if isinstance(v, Tensor)}  # type: ignore

    @property
    def is_nested(self) -> bool:
        """Returns whether episodes have different lengths (if the tensors are nested tensors)."""
        return self.observations.is_nested

    @property
    def episode_lengths(self) -> list[int]:
        """The number of steps in each episode in the batch."""
        return [len(ep_infos) for ep_infos in self.infos]

    @property
    def batch_size(self) -> int:
        """The number of episodes in the batch."""
        return self.terminated.shape[0]

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode[ActorOutput]]) -> EpisodeBatch[ActorOutput]:
        """Collates a list of episodes into an EpisodeBatch object containing (possibly nested)
        tensors."""
        rewards = [ep.rewards for ep in episodes]
        from project.datamodules.rl.stacking_utils import stack
        from project.datamodules.rl.wrappers.jax_torch_interop import (
            jax_to_torch_tensor,
        )

        stacked_rewards = stack(rewards)  # type: ignore
        assert isinstance(stacked_rewards, np.ndarray | jax.Array | torch.Tensor)
        if isinstance(stacked_rewards, jax.Array):
            stacked_rewards = jax_to_torch_tensor(stacked_rewards)
        if isinstance(stacked_rewards, np.ndarray):
            stacked_rewards = torch.as_tensor(stacked_rewards)

        terminated = stack([ep.terminated for ep in episodes])
        truncated = stack([ep.terminated for ep in episodes])
        assert isinstance(terminated, torch.Tensor | jax.Array | np.ndarray), terminated
        assert isinstance(truncated, torch.Tensor | jax.Array | np.ndarray), truncated
        if isinstance(terminated, jax.Array):
            terminated = jax_to_torch_tensor(terminated)
        if isinstance(terminated, np.ndarray):
            terminated = torch.as_tensor(terminated, dtype=torch.bool)

        if isinstance(truncated, jax.Array):
            truncated = jax_to_torch_tensor(truncated)
        if isinstance(truncated, np.ndarray):
            truncated = torch.as_tensor(truncated, dtype=torch.bool)

        final_observation: torch.Tensor | None = None
        final_observations = [ep.final_observation for ep in episodes]
        if is_list_of(final_observations, Tensor):
            final_observation = stack(final_observations)

        final_infos: list[dict] | None = None
        if any(ep.final_info is not None for ep in episodes):
            assert all(ep.final_info is not None for ep in episodes)
            _final_infos = [ep.final_info for ep in episodes]
            assert is_list_of(_final_infos, dict)
            final_infos = _final_infos

        return EpisodeBatch(
            observations=stack([ep.observations for ep in episodes]),
            actions=stack([ep.actions for ep in episodes]),
            rewards=stacked_rewards,
            # TODO: Could perhaps stack the infos so it mimics what the RecordEpisodeStatistics wrapper
            # does for VectorEnvs.
            infos=[ep.infos for ep in episodes],
            terminated=terminated,
            truncated=truncated,
            actor_outputs=stack([ep.actor_outputs for ep in episodes]),
            final_infos=final_infos,
            final_observations=final_observation,
        )

    def split(self) -> list[Episode[ActorOutput]]:
        """Splits the batch into a list of episodes."""

        def _unstack(values):
            if isinstance(values, torch.Tensor):
                return values.unbind(0)
            if isinstance(values, dict):
                return {k: _unstack(v) for k, v in values.items()}
            raise NotImplementedError(f"Cannot unstack values {values}.")

        return [
            Episode(
                observations=obs,
                actions=act,
                rewards=rew,
                infos=info,
                terminated=term,
                truncated=trunc,
                actor_outputs=act_out,
                final_observation=final_obs,
                final_info=final_info,
            )
            for obs, act, rew, info, term, trunc, act_out, final_obs, final_info in zip(
                self.observations.unbind(0),
                self.actions.unbind(0),
                self.rewards.unbind(0),
                self.infos,
                self.terminated.unbind(0),
                self.truncated.unbind(0),
                _unstack(self.actor_outputs),
                (
                    self.final_observations.unbind(0)
                    if isinstance(self.final_observations, Tensor)
                    else [None] * self.batch_size
                ),
                (self.final_infos if self.final_infos is not None else [None] * self.batch_size),
            )
        ]


class EpisodeInfo(TypedDict):
    """Shows what the `gym.wrappers.RecordEpisodeStatistics` wrapper stores in the step's info."""

    episode: NotRequired[EpisodeStats]
    """Statistics that are stored by the wrapper at the end of an episode.

    This entry is only present for the last step in the episode.
    """


class EpisodeStats(TypedDict):
    r: float
    """Cumulative reward."""

    l: int  # noqa
    """Episode length."""

    t: float
    """Elapsed time since instantiation of wrapper."""


# TODO: These are unused at the moment, but once we use VectorEnvs (if we do), then we'll need to
# change the Info type on the EpisodeBatch object since this is what the `RecordEpisodeStatistics`
# wrapper adds to the `info` dictionary.
class VectorEnvEpisodeInfos(TypedDict, total=False):
    """What the `gym.wrappers.RecordEpisodeStatistics` wrapper stores in a VectorEnv's info."""

    _episode: np.ndarray
    """Boolean array of length num-envs."""
    episode: VectorEnvEpisodeStats


class VectorEnvEpisodeStats(TypedDict):
    r: np.ndarray
    """Cumulative rewards for each env."""

    l: int  # noqa
    """Episode lengths for each env."""

    t: float
    """Elapsed time since instantiation of wrapper for each env."""

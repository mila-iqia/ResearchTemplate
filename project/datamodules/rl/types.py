from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, NotRequired, TypedDict

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

from project.utils.types import NestedDict, NestedMapping  # noqa

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

    rewards: Tensor | np.ndarray | jax.Array
    # todo: wrappers assume that rewards is a numpy array for now. Might be easy to use jax Arrays instead of torch Tensors here.

    infos: list[dict]
    truncated: bool | jax.numpy.bool_ | torch.BoolTensor
    terminated: bool | jax.numpy.bool_ | torch.BoolTensor
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
        return tuple(
            MiddleTransition(
                observation=self.observations[i],
                info=self.infos[i],
                action=self.actions[i],
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
                reward=self.rewards[-1],
                terminated=self.terminated,
                truncated=self.truncated,
                final_observation=self.final_observation,
                final_info=self.final_info,
            ),
        )


@dataclasses.dataclass(frozen=True)
class Transition:
    observation: Tensor
    info: dict
    action: Tensor
    reward: Tensor | jax.Array


@dataclasses.dataclass(frozen=True)
class MiddleTransition(Transition):
    next_observation: Tensor
    next_info: dict
    is_terminal: ClassVar[bool] = False


@dataclasses.dataclass(frozen=True)
class FinalTransition(Transition):
    terminated: bool
    truncated: bool
    final_observation: Tensor | None = None
    final_info: dict | None = None
    is_terminal: ClassVar[bool] = True


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

    observations: Tensor  # [num_envs (a.k.a. batch_size), <episode_length>, *<observation_shape>]
    actions: Tensor  # [num_envs (a.k.a. batch_size), <episode_length>, *<action_shape>]
    rewards: Tensor  # [num_envs (a.k.a. batch_size), <episode_length>]

    infos: list[list[EpisodeInfo]]
    """List of the `info` dictionaries at each step, for each episode in the batch."""

    truncated: Tensor
    """Whether the episode was truncated (i.e. a maximum number of steps was reached)."""
    terminated: Tensor
    """Whether the episode ended "normally"."""

    actor_outputs: ActorOutput
    """Additional outputs of the Actor (besides the action to take) for each step in the episode.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """


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

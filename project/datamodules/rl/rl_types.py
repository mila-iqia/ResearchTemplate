from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import gym
import gym.spaces
import gymnasium.spaces
import numpy as np
from gym import Space
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import Generic, TypeVar  # noqa

type _Env[ObsType, ActType] = gym.Env[ObsType, ActType] | gymnasium.Env[ObsType, ActType]
type _Space[T_cov] = gym.Space[T_cov] | gymnasium.Space[T_cov]

BoxSpace = gym.spaces.Box | gymnasium.spaces.Box
DiscreteSpace = gym.spaces.Discrete | gymnasium.spaces.Discrete


TensorType = TypeVar("TensorType", bound=Tensor, default=Tensor)
ObservationT = TypeVar("ObservationT", default=np.ndarray)
ActionT = TypeVar("ActionT", default=int)
# TypeVars for the observations and action types for a gym environment.

ActorOutput = TypeVar("ActorOutput", default=dict, covariant=True)

Actor = Callable[[ObservationT, Space[ActionT]], tuple[ActionT, ActorOutput]]
"""An "Actor" is given an observation and action space, and asked to produce the next action.

It can also output other stuff that will be used to train the model later.
"""

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


### Typing fixes for gymnasium.

# TODO: annoying typing thing with gymnasium.Env.[observation|action]_space: The type is always
# Space[ActType], even if you manually set it to something else, because of the property (which in
# itself is useless!)


# gym.Env subclasses typing.Generic which atm doesn't allow default typevars
class Env[ObsType, ActType](gym.Env[ObsType, ActType]):
    observation_space: Space[ObsType]
    action_space: Space[ActType]


# VectorEnv doesn't have type hints in current gymnasium.
class VectorEnv[ObsType, ActType](gymnasium.vector.VectorEnv, Env[ObsType, ActType]):
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


# gymnasium.Wrapper doesn't have default types for the ObsType and ActType which makes it more difficult to use.
class Wrapper(
    gymnasium.Wrapper[WrapperObsType, WrapperActType, WrappedObsType, WrappedActType],
    Generic[WrapperObsType, WrapperActType, WrappedObsType, WrappedActType],
):
    def __init__(self, env: gymnasium.Env[WrappedObsType, WrappedActType]):
        super().__init__(env)
        # Also, for some reason, it has a @property for these? but getattr/setattr already works fine.
        self.observation_space: Space[WrapperObsType]
        self.action_space: Space[WrapperActType]


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


def random_actor(observation: Any, action_space: Space[ActionT]) -> tuple[ActionT, dict]:
    """Actor that takes random actions."""
    return action_space.sample(), {}


# todo: make this generic w.r.t. obs / act types once we add envs with more complicated observations.
class Episode(TypedDict, Generic[ActorOutput]):
    """An Episode where the contents have been stacked into tensors instead of kept in lists."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    infos: list[EpisodeInfo]
    truncated: bool
    terminated: bool
    actor_outputs: ActorOutput


class UnstackedEpisode(TypedDict, Generic[ActorOutput]):
    """Data of a (possibly on-going) episode, where the contents haven't yet been stacked."""

    observations: list[Tensor]
    actions: list[Tensor]
    rewards: list[float]
    infos: list[EpisodeInfo]

    truncated: bool
    """Whether the episode was truncated (i.e. a maximum number of steps was reached)."""
    terminated: bool
    """Whether the episode ended "normally"."""

    actor_outputs: list[ActorOutput]
    """Additional outputs of the Actor (besides the action to take) for each step in the episode.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """


class EpisodeBatch(TypedDict, Generic[ActorOutput]):
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

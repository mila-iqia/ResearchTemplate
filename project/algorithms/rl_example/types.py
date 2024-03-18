from __future__ import annotations

from typing import Any, Callable, Generic
from gym import Space
import numpy as np
from torch import Tensor
from typing_extensions import TypeVar, TypedDict, ParamSpec, NotRequired


TensorType = TypeVar("TensorType", bound=Tensor, default=Tensor)
ObservationT = TypeVar("ObservationT", default=np.ndarray)
ActionT = TypeVar("ActionT", default=int)
D = TypeVar("D", bound=TypedDict)
K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
P = ParamSpec("P")
# TypeVars for the observations and action types for a gym environment.


class ActorOutput(TypedDict):
    """Additional outputs of the Actor (besides the action to take) for a single step in the env.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """

    # For example:
    # action_log_probabilities: Tensor


ActorOutputDict = TypeVar(
    "ActorOutputDict", bound=ActorOutput, default=ActorOutput, covariant=True
)

Actor = Callable[[ObservationT, Space[ActionT]], tuple[ActionT, ActorOutputDict]]
"""An "Actor" is given an observation and action space, and asked to produce the next action.

It can also output other stuff that will be used to train the model later.
"""


def random_actor(observation: Any, action_space: Space[ActionT]) -> tuple[ActionT, ActorOutput]:
    """Actor that takes random actions."""
    return action_space.sample(), {}


class Episode(TypedDict, Generic[ActorOutputDict]):
    """An Episode where the contents have been stacked into tensors instead of kept in lists."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    infos: list[EpisodeInfo]
    truncated: bool
    terminated: bool
    actor_outputs: ActorOutputDict


class UnstackedEpisode(TypedDict, Generic[ActorOutputDict]):
    """Data of a (possibly on-going) episode, where the contents haven't yet been stacked."""

    observations: list[Tensor]
    actions: list[Tensor]
    rewards: list[float]
    infos: list[EpisodeInfo]

    truncated: bool
    """Whether the episode was truncated (i.e. a maximum number of steps was reached)."""
    terminated: bool
    """Whether the episode ended "normally"."""

    actor_outputs: list[ActorOutputDict]
    """Additional outputs of the Actor (besides the action to take) for each step in the episode.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """


class EpisodeBatch(TypedDict, Generic[ActorOutputDict]):
    """Object that contains a batch of episodes, stored as nested tensors.

    A [nested tensor](https://pytorch.org/docs/stable/nested.html#module-torch.nested) is basically
    a list of tensors, where the stored tensors may have different lengths.

    In our case, this is useful since episodes might have different number of steps, and it is more
    efficient to perform a single forward pass with a nested tensor than it would be to do a python
    for loop with multiple forward passes for each episode.

    See https://pytorch.org/docs/stable/nested.html#module-torch.nested for more info.
    """

    observations: Tensor  # [batch_size, <episode_length>, *<observation_shape>]
    actions: Tensor  # [batch_size, <episode_length>, *<action_shape>]
    rewards: Tensor  # [batch_size, <episode_length>]

    infos: list[list[EpisodeInfo]]
    """List of the `info` dictionaries at each step, for each episode in the batch."""

    truncated: Tensor
    """Whether the episode was truncated (i.e. a maximum number of steps was reached)."""
    terminated: Tensor
    """Whether the episode ended "normally"."""

    actor_outputs: ActorOutputDict
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

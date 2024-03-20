from __future__ import annotations

import itertools
from typing import Any, Generic, Unpack, overload

import gym
import gym.spaces
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .types import Actor, ActorOutputDict, Episode, UnstackedEpisode
from .utils import stack_dicts

eps = np.finfo(np.float32).eps.item()


def stack_episode(
    **episode: Unpack[UnstackedEpisode[ActorOutputDict]],
) -> Episode[ActorOutputDict]:
    """Stacks the lists of items at each step into an Episode dict containing tensors."""
    device = _get_device(episode)
    return Episode(
        observations=_stack(episode["observations"]),
        actions=_stack(episode["actions"]),
        rewards=torch.as_tensor(episode["rewards"], device=device),
        infos=episode["infos"],
        truncated=episode["truncated"],
        terminated=episode["terminated"],
        actor_outputs=_stack(episode["actor_outputs"]),
    )


class RlDataset(IterableDataset[Episode[ActorOutputDict]], Generic[ActorOutputDict]):
    def __init__(
        self,
        env: gym.Env[Tensor, Tensor],
        actor: Actor[Tensor, Tensor, ActorOutputDict],
        episodes_per_epoch: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.env = env
        self.actor = actor
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self._episode_index = 0

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.env})"

    # def __next__(self) -> Episode[ObsType, ActType, ActorOutputType]:
    #     self._episode_index += 1
    #     if self._episode_index == self.episodes_per_epoch:
    #         raise StopIteration
    #     return self.collect_one_episode()

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        for _episode_index in range(self.episodes_per_epoch):
            yield self.collect_one_episode(seed=self.seed if _episode_index == 0 else None)

    def collect_one_episode(self, seed: int | None = None) -> Episode[ActorOutputDict]:
        # logger.debug(f"Starting episode {self._episode_index}.")
        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards = []
        infos = []
        actor_outputs: list[ActorOutputDict] = []
        terminated = False
        truncated = False

        obs, info = self.env.reset(seed=seed)
        observations.append(obs)
        infos.append(info)

        # note: assuming a TimeLimit wrapper is present, otherwise the episode could last forever.
        for _episode_step in itertools.count():
            action, actor_output = self.actor(obs, self.env.action_space)
            actions.append(action)
            actor_outputs.append(actor_output)

            obs, reward, terminated, truncated, info = self.env.step(action)

            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if terminated or truncated:
                break

        return stack_episode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            truncated=truncated,
            terminated=terminated,
            actor_outputs=actor_outputs,
        )


@overload
def _stack(values: list[Tensor], **kwargs) -> Tensor: ...


@overload
def _stack(values: list[ActorOutputDict], **kwargs) -> ActorOutputDict: ...


def _stack(values: list[Tensor] | list[ActorOutputDict], **kwargs) -> Tensor | ActorOutputDict:
    """Stack lists of tensors into a Tensor or dicts of tensors into a dict of stacked tensors."""
    if isinstance(values[0], dict):
        # NOTE: weird bug in the type checker? Doesn't understand that `values` is a list[Tensor].
        return stack_dicts(values, **kwargs)  # type: ignore
    return torch.stack(values, **kwargs)  # type: ignore


def _get_device(values: Any) -> torch.device:
    """Retrieve the Device of the first found Tensor in `values`."""

    def _get_device(value: Tensor | Any) -> torch.device | None:
        if isinstance(value, Tensor):
            return value.device
        if isinstance(value, dict):
            for k, v in value.items():
                device = _get_device(v)
                if device is not None:
                    return device
            return None
        if isinstance(value, list | tuple):
            for v in value:
                device = _get_device(v)
                if device is not None:
                    return device
            return None
        return None

    device = _get_device(values)
    if device is None:
        raise ValueError("There are no tensors in values, can't determine the device!")
    return device

from __future__ import annotations

import itertools
from typing import Generic

import gym
import gym.spaces
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from .rl_types import Actor, ActorOutput, Episode
from .stacking_utils import stack_episode

eps = np.finfo(np.float32).eps.item()


class RlDataset(IterableDataset[Episode[ActorOutput]], Generic[ActorOutput]):
    def __init__(
        self,
        env: gym.Env[Tensor, Tensor],
        actor: Actor[Tensor, Tensor, ActorOutput],
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

    def collect_one_episode(self, seed: int | None = None) -> Episode[ActorOutput]:
        # logger.debug(f"Starting episode {self._episode_index}.")
        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards = []
        infos = []
        actor_outputs: list[ActorOutput] = []
        terminated = False
        truncated = False

        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        obs, info = self.env.reset(seed=seed)

        observations.append(obs)
        infos.append(info)

        # note: assuming a TimeLimit wrapper is present, otherwise the episode could last forever.
        for _episode_step in itertools.count():
            action, actor_output = self.actor(obs, self.env.action_space)

            obs, reward, terminated, truncated, info = self.env.step(action)

            actions.append(action)
            actor_outputs.append(actor_output)
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

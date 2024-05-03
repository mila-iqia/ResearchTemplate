from __future__ import annotations

import itertools
from typing import Generic

import gym
import gym.spaces
import gym.vector
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from .rl_types import Actor, ActorOutput, Episode, EpisodeInfo
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


class VectorEnvRlDataset(IterableDataset[Episode[ActorOutput]], Generic[ActorOutput]):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        actor: Actor[Tensor, Tensor, ActorOutput],
        episodes_per_epoch: int,
        seed: int | list[int] | None = None,
    ):
        super().__init__()
        self.env = env
        self.actor = actor
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self._episode_index = 0
        self.num_envs = env.num_envs

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"env={self.env}, "
            f"actor={self.actor}, "
            f"episodes_per_epoch={self.episodes_per_epoch}, "
            f"seed={self.seed}"
            f")"
        )

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        observations: list[list[Tensor]] = [[] for _ in range(self.num_envs)]
        actions: list[list[Tensor]] = [[] for _ in range(self.num_envs)]
        rewards: list[list[float]] = [[] for _ in range(self.num_envs)]
        infos: list[list[EpisodeInfo]] = [[] for _ in range(self.num_envs)]
        actor_outputs: list[list[ActorOutput]] = [[] for _ in range(self.num_envs)]

        obs_batch = self.env.reset(seed=self.seed)
        for i, obs in enumerate(obs_batch):
            observations[i].append(obs)

        while self._episode_index < self.episodes_per_epoch:
            action_batch, actor_output_batch = self.actor(obs_batch, self.env.action_space)

            _step_output = self.env.step(action_batch)
            assert _step_output is not None
            assert isinstance(_step_output, tuple) and len(_step_output) == 5
            obs_batch, reward_batch, done_batch, _truncated_batch, info_batch = _step_output  # type: ignore

            for env_index, (
                obs,
                reward,
                done,
                _truncated,
                info,
                action,
                actor_output,
            ) in enumerate(
                zip(
                    obs_batch,
                    reward_batch,
                    done_batch,
                    _truncated_batch,
                    info_batch,
                    action_batch,
                    actor_output_batch,
                )
            ):
                if done:
                    yield stack_episode(
                        observations=observations[env_index],
                        actions=actions[env_index],
                        rewards=rewards[env_index],
                        infos=infos[env_index],
                        truncated=_truncated,
                        terminated=done,
                        actor_outputs=actor_outputs[env_index],
                    )

                    observations[env_index].clear()
                    actions[env_index].clear()
                    rewards[env_index].clear()
                    infos[env_index].clear()
                    actor_outputs[env_index].clear()

                    self._episode_index += 1
                    if self._episode_index == self.episodes_per_epoch:
                        break

                # todo: double check that the obs is the first of the new episode, and not the last
                # of the previous one. Also check the timing of the action so it fits in the right
                # episode.
                observations[env_index].append(obs)
                actions[env_index].append(action)
                rewards[env_index].append(reward)
                infos[env_index].append(info)
                actor_outputs.append(actor_output)

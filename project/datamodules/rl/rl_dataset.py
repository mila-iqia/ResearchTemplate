from __future__ import annotations

import collections.abc
import itertools
from collections.abc import Iterable, Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Generic

import gymnasium
import jax.experimental.compilation_cache.compilation_cache
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from project.utils.types import NestedDict, NestedMapping

from .rl_types import Actor, ActorOutput, Episode, EpisodeInfo, VectorEnv
from .stacking_utils import stack_episode

logger = get_logger(__name__)
eps = np.finfo(np.float32).eps.item()

jax.experimental.compilation_cache.compilation_cache.set_cache_dir(Path.home() / ".cache/jax")


class RlDataset(IterableDataset[Episode[ActorOutput]]):
    def __init__(
        self,
        env: gymnasium.Env[Tensor, Tensor],
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

    def __iter__(self) -> Iterable[Episode[ActorOutput]]:
        for _episode_index in range(self.episodes_per_epoch):
            episode = self.collect_one_episode(seed=self.seed if _episode_index == 0 else None)
            logger.info(
                f"Finished episode {_episode_index} which lasted {episode.observations.shape[0]} steps."
            )
            logger.debug("Total rewards: {}", episode.rewards.sum())
            yield episode

    def collect_one_episode(self, seed: int | None = None) -> Episode[ActorOutput]:
        # logger.debug(f"Starting episode {self._episode_index}.")
        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards = []
        infos = []
        actor_outputs: list[ActorOutput] = []
        terminated = False
        truncated = False

        if seed is not None:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
        obs, info = self.env.reset(seed=seed)

        observations.append(obs)
        infos.append(info)

        # note: assuming a TimeLimit wrapper is present, otherwise the episode could last forever.
        for _episode_step in itertools.count():
            action, actor_output = self.actor(obs, self.env.action_space)
            obs, reward, terminated, truncated, info = self.env.step(action)
            logger.debug(f"step {_episode_step}, {{}}", terminated)

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
        env: VectorEnv[Tensor, Tensor],
        actor: Actor[Tensor, Tensor, ActorOutput],
        episodes_per_epoch: int | None = None,
        steps_per_epoch: int | None = None,
        seed: int | list[int] | None = None,
    ):
        super().__init__()
        self.env = env
        self.actor = actor
        self.episodes_per_epoch = episodes_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

        self.num_envs = env.num_envs
        # TODO: Need to call a method on the iterator to reset the envs when the actor is updated.
        self._iterator: Iterable[Episode[ActorOutput]] | None = None

    def on_actor_update(self) -> None:
        """Call this when the actor neural net has its weights updated so the iterator resets the
        envs (to prevent having a mix of old and new actions in any given episode)"""

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"env={self.env}, "
            f"actor={self.actor}, "
            f"episodes_per_epoch={self.episodes_per_epoch}, "
            f"seed={self.seed}"
            f")"
        )

    def __iter__(self) -> Iterable[Episode[ActorOutput]]:
        # Reset the env RNG at each epoch start? or only on the first epoch?
        yield from VectorEnvEpisodeIterator(
            env=self.env,
            actor=self.actor,
            initial_seed=self.seed,
            max_episodes=self.episodes_per_epoch,
            max_steps=self.steps_per_epoch,
        )

    def __len__(self) -> int:
        assert self.episodes_per_epoch is not None
        return self.episodes_per_epoch


def VectorEnvEpisodeIterator[ActorOutput: NestedMapping[str, Tensor]](
    env: VectorEnv[Tensor, Tensor],
    actor: Actor[Tensor, Tensor, ActorOutput],
    initial_seed: int | list[int] | None = None,
    max_episodes: int | None = None,
    max_steps: int | None = None,
) -> Iterable[Episode[ActorOutput]]:
    """Iterator that yields one episode at a time from a VectorEnv."""
    logger.debug(f"Making a new iterator for {env=}.")
    num_envs = env.num_envs

    observations: list[list[Tensor]] = [[] for _ in range(num_envs)]
    actions: list[list[Tensor]] = [[] for _ in range(num_envs)]
    rewards: list[list[Tensor]] = [[] for _ in range(num_envs)]
    infos: list[list[EpisodeInfo]] = [[] for _ in range(num_envs)]
    actor_outputs: list[list[ActorOutput]] = [[] for _ in range(num_envs)]

    obs_batch, info_batch = env.reset(seed=initial_seed)

    # Note: not really used atm since we assume the env gives back tensors (not dicts)
    # if isinstance(obs_batch, Mapping):
    #     obs_batch = list(sliced_dict(obs_batch, n_slices=num_envs))
    for i, env_obs in enumerate(obs_batch):
        observations[i].append(env_obs)

    info_batch = list(sliced_dict(info_batch, n_slices=num_envs))
    for i, env_info in enumerate(info_batch):
        infos[i].append(env_info)  # type: ignore

    _steps: int = 0
    _episodes: int = 0

    should_stop = False
    while not should_stop:
        action_batch, actor_output_batch = actor(obs_batch, env.action_space)
        logger.debug(
            f"Step {_steps}: # episode lengths in each env: {[len(obs) for obs in observations]}"
        )

        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = env.step(
            action_batch
        )
        env_infos = list(sliced_dict(info_batch, n_slices=num_envs))
        env_actor_outputs = list(sliced_dict(actor_output_batch, n_slices=num_envs))

        for env_index in range(num_envs):
            _steps += 1
            env_obs = obs_batch[env_index]
            env_reward = reward_batch[env_index]
            env_terminated = terminated_batch[env_index]
            env_truncated = truncated_batch[env_index]
            env_info: EpisodeInfo = env_infos[env_index]  # type: ignore
            env_action = action_batch[env_index]
            env_actor_output = env_actor_outputs[env_index]

            if env_terminated | env_truncated:
                _episodes += 1

                # The observation and info are of the first step of the next episode.
                # The action and reward are from the last step of the previous episode.
                rewards[env_index].append(env_reward)
                actions[env_index].append(env_action)
                actor_outputs[env_index].append(env_actor_output)

                # We don't really use these as far as I can tell (the jax envs don't produce them).
                final_observation: Tensor | None = env_info.get("old_observation")
                final_info: EpisodeInfo | None = env_info.get("final_info")

                episode = stack_episode(
                    observations=observations[env_index],
                    actions=actions[env_index],
                    rewards=rewards[env_index],
                    infos=infos[env_index].copy(),
                    terminated=env_terminated,
                    truncated=env_truncated,
                    actor_outputs=actor_outputs[env_index],
                    final_observation=final_observation,
                    final_info=final_info,
                )
                logger.debug(
                    f"Episode {_episodes} is done (contains {episode.observations.shape[0]} steps)"
                )
                yield episode

                observations[env_index].clear()
                rewards[env_index].clear()
                infos[env_index].clear()
                actions[env_index].clear()
                actor_outputs[env_index].clear()

                # The observation and info are of the first step of the next episode.
                observations[env_index].append(env_obs)
                infos[env_index].append(env_info)

                if _episodes == max_episodes:
                    # Break now, so we return the right number of episodes even if multiple envs
                    # finished at the same time on the last step.
                    logger.info(f"Reached episode limit ({max_episodes}), exiting.")
                    should_stop = True  # exit outer loop
                    break  # exit inner loop
            else:
                observations[env_index].append(env_obs)
                rewards[env_index].append(env_reward)
                infos[env_index].append(env_info)
                actions[env_index].append(env_action)
                actor_outputs[env_index].append(env_actor_output)

            if _steps == max_steps:
                logger.info(f"Reached limit on the total number of steps ({max_steps}), exiting.")
                should_stop = True  # exit outer loop.
                break

    logger.info(f"Finished an epoch after {_steps} steps and {_episodes} episodes.")


def sliced_dict[M: NestedMapping[str, Tensor | None]](
    d: M, n_slices: int | None = None
) -> Iterable[M]:
    """Slice a dict of sequences (tensors) into a sequence of dicts with the values at the same
    index in the 0th dimension."""
    if not d:
        assert n_slices is not None
        for _ in range(n_slices):
            yield {}
        return

    def get_len(d: NestedMapping[str, Tensor | None]):
        length: int | None = None
        for k, v in d.items():
            if isinstance(v, Mapping):
                length = get_len(v)
                continue
            if v is None:
                continue
            if length is None:
                length = len(v)
            else:
                assert length == len(v)
        return length

    if n_slices is None:
        n_slices = get_len(d)
        assert n_slices is not None

    def _sliced(
        d: NestedMapping[str, Tensor | None], index: int
    ) -> NestedDict[str, Tensor | None]:
        result: NestedDict[str, Tensor | None] = {}
        for k, v in d.items():
            if isinstance(v, Mapping):
                result[k] = _sliced(v, index)
            elif v is None:
                result[k] = None
            elif (
                isinstance(v, collections.abc.Sequence | Tensor | np.ndarray)
                and len(v) == n_slices
            ):
                result[k] = v[index]
            elif isinstance(v, int | float | bool | str):
                # Copy the value at every index, for instance if the actor returns a single int for
                # a batch of observations
                result[k] = v
            else:
                raise NotImplementedError(
                    f"Don't know how to slice value {v} at key {k} from the actor output dict {d}"
                )
        return result

    for i in range(n_slices):
        yield _sliced(d, i)
    return

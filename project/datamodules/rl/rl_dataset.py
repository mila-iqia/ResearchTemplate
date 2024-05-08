from __future__ import annotations

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


def VectorEnvEpisodeIterator(
    env: VectorEnv[Tensor, Tensor],
    actor: Actor[Tensor, Tensor, ActorOutput],
    initial_seed: int | list[int] | None = None,
    max_episodes: int | None = None,
    max_steps: int | None = None,
):
    """Iterator that yields one episode at a time from a VectorEnv."""
    logger.debug(f"Making a new iterator for {env=}.")
    num_envs = env.num_envs

    observations: list[list[Tensor]] = [[] for _ in range(num_envs)]
    actions: list[list[Tensor]] = [[] for _ in range(num_envs)]
    rewards: list[list[float]] = [[] for _ in range(num_envs)]
    infos: list[list[EpisodeInfo]] = [[] for _ in range(num_envs)]
    actor_outputs: list[list[ActorOutput]] = [[] for _ in range(num_envs)]
    _episodes: int = 0
    _steps: int = 0

    obs_batch, info_batch = env.reset(seed=initial_seed)
    for i, env_obs in enumerate(obs_batch):
        observations[i].append(env_obs)
    for i, env_info in enumerate(sliced_dict(info_batch, n_slices=num_envs)):
        infos[i].append(env_info)

    should_stop = False

    while not should_stop:
        action_batch, actor_output_batch = actor(obs_batch, env.action_space)
        logger.debug(
            f"Step {_steps}: # episode lengths in each env: {[len(obs) for obs in observations]}"
        )

        obs_batch, reward_batch, done_batch, truncated_batch, info_batch = env.step(action_batch)

        for (
            env_index,
            env_info,
            env_actor_output,
        ) in itertools.zip_longest(
            range(num_envs),
            sliced_dict(info_batch, n_slices=num_envs),
            sliced_dict(actor_output_batch, n_slices=num_envs),
            fillvalue=None,
        ):
            assert env_index is not None
            assert env_info is not None
            assert env_actor_output is not None

            env_obs = obs_batch[env_index]
            env_reward = reward_batch[env_index]
            env_done = done_batch[env_index]
            env_truncated = truncated_batch[env_index]
            env_action = action_batch[env_index]

            if env_done:
                episode = stack_episode(
                    observations=observations[env_index],
                    actions=actions[env_index],
                    rewards=rewards[env_index],
                    infos=infos[env_index],
                    terminated=env_done,
                    truncated=env_truncated,
                    actor_outputs=actor_outputs[env_index],
                )
                # TODO: turn back down to debug level after testing is done.
                logger.info(
                    f"Episode {_episodes} is done (contains {episode.observations.shape[0]} steps)"
                )
                yield episode

                observations[env_index].clear()
                actions[env_index].clear()
                rewards[env_index].clear()
                infos[env_index].clear()
                actor_outputs[env_index].clear()

                _episodes += 1
                if _episodes == max_episodes:
                    _steps += 1  # so we exit the outer loop with the right number of total steps.
                    should_stop = True  # so we exit the outer loop.
                    break

            # note: This is done after the episode yielding so that we yield the right number of
            # episodes.
            _steps += 1
            # todo: test for off-by-1 errors. For example, if the episode length is fixed to 200,
            # then if the max steps is 799, we expect to see 3 episodes, not 4.
            if max_steps and _steps == max_steps - 1:
                should_stop = True  # so we exit the outer loop.
                break

            # todo: double check that the obs is the first of the new episode, and not the last
            # of the previous one. Also check the timing of the action so it fits in the right
            # episode.
            observations[env_index].append(env_obs)
            actions[env_index].append(env_action)
            rewards[env_index].append(env_reward)
            infos[env_index].append(env_info)  # type: ignore
            actor_outputs[env_index].append(env_actor_output)  # type: ignore
    logger.info(f"Finished an epoch after {_steps} steps and {_episodes} episodes.")


def sliced_dict(
    d: NestedMapping[str, Tensor | None], n_slices: int | None = None
) -> Iterable[NestedDict[str, Tensor | None]]:
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
            else:
                result[k] = v[index]
        return result

    for i in range(n_slices):
        yield _sliced(d, i)
    return

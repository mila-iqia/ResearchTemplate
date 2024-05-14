from __future__ import annotations

import collections.abc
import itertools
from collections.abc import Iterable, Iterator, Mapping
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

    def on_actor_update(self) -> None:
        pass  # do nothing, no buffers to clear or anything like that, really.

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self) -> Iterable[Episode[ActorOutput]]:
        for _episode_index in range(self.episodes_per_epoch):
            episode = self.collect_one_episode(seed=self.seed if _episode_index == 0 else None)
            logger.info(
                f"Finished episode {_episode_index} which lasted {episode.observations.shape[0]} steps."
            )
            logger.debug("Total rewards: %d", episode.rewards.sum())
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
            logger.debug(f"step {_episode_step}, %s", terminated)

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
        self._actor = actor
        self.episodes_per_epoch = episodes_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

        self.num_envs = env.num_envs
        # TODO: Need to call a method on the iterator to reset the envs when the actor is updated.
        self._iterator: VectorEnvEpisodeIterator[ActorOutput] | None = None

    @property
    def actor(self) -> Actor[Tensor, Tensor, ActorOutput]:
        return self._actor

    @actor.setter
    def actor(self, value: Actor[Tensor, Tensor, ActorOutput]) -> None:
        self._actor = value
        self.on_actor_update()

    def on_actor_update(self) -> None:
        """Call this after updating the weights of the actor neural net, so the env iterator can
        reset the environments.

        This is done to prevent having a mix of old and new actions in any given episode, which
        would cause errors when backpropagating.
        """
        if self._iterator is None:
            logger.warning("The actor has been updated before starting to iterate on the env?")
            return

        logger.debug(
            f"The actor has been updated at step {self._iterator._yielded_steps} "
            f"(episode {self._iterator._yielded_episodes}). Resetting the envs."
        )
        self._iterator.actor = self.actor
        self._iterator.reset_envs()

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
        if self._iterator is None:
            self._iterator = VectorEnvEpisodeIterator(
                env=self.env,
                actor=self.actor,
                initial_seed=self.seed,
                max_episodes=self.episodes_per_epoch,
                max_steps=self.steps_per_epoch,
            )
        yield from iter(self._iterator)

    def __len__(self) -> int:
        assert self.episodes_per_epoch is not None
        return self.episodes_per_epoch


class VectorEnvEpisodeIterator[ActorOutput: NestedMapping[str, Tensor]](
    Iterator[Episode[ActorOutput]]
):
    """Yields one episode at a time from interacting with a vectorized environment."""

    def __init__(
        self,
        env: VectorEnv[Tensor, Tensor],
        actor: Actor[Tensor, Tensor, ActorOutput],
        initial_seed: int | list[int] | None = None,
        max_episodes: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Iterator that yields one episode at a time from a VectorEnv."""
        logger.debug(f"Making a new iterator for {env=}.")
        self.num_envs = env.num_envs
        self.env = env
        self.initial_seed = initial_seed
        self.actor = actor
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.observations: list[list[Tensor]] = [[] for _ in range(self.num_envs)]
        self.rewards: list[list[Tensor]] = [[] for _ in range(self.num_envs)]
        self.infos: list[list[dict]] = [[] for _ in range(self.num_envs)]
        self.actions: list[list[Tensor]] = [[] for _ in range(self.num_envs)]
        self.actor_outputs: list[list[ActorOutput]] = [[] for _ in range(self.num_envs)]

        self._yielded_steps: int = 0
        self._yielded_episodes: int = 0

        # The last observation and info from the env. The last observation is fed to the actor and
        # updated at each step.
        self._last_observation: Tensor | None = None
        self._last_info: dict | None = None
        self._episodes_to_yield_at_this_step: list[Episode[ActorOutput]] = []

        self.reset_envs(seed=initial_seed)

    def __next__(self) -> Episode[ActorOutput]:
        """Fetch the next completed episode, possibly taking multiple steps in the environment."""
        if self._should_stop:
            if self._yielded_episodes == self.max_episodes:
                logger.info(f"Reached limit of {self.max_episodes} episodes.")
            if self.max_steps and self._yielded_steps >= self.max_steps:
                logger.info(f"Reached limit of {self.max_steps} steps.")
            raise StopIteration

        # Note: no need to check `self._should_stop` here since we only count the yielded steps and
        # episodes, not the steps in the environment.
        while not self._episodes_to_yield_at_this_step:
            # Take a step and store the completed episodes (if any).
            self._episodes_to_yield_at_this_step = self.step_and_yield_completed_episodes()
        episode = self._episodes_to_yield_at_this_step.pop(0)
        self._yielded_episodes += 1
        self._yielded_steps += episode.length
        return episode

    def __iter__(self):
        return self

        # while not self._should_stop:
        #     for episode in self.step_and_yield_completed_episodes():
        #         yield episode
        #         self._yielded_episodes += 1
        #         self._yielded_steps += episode.length

        #         if self._should_stop:
        #             if self._yielded_episodes == self.max_episodes:
        #                 logger.info(f"Reached limit of {self.max_episodes} episodes.")
        #             if self.max_steps and self._yielded_steps >= self.max_steps:
        #                 logger.info(f"Reached limit of {self.max_steps} steps.")

        #             # Break now, so we return the right number of episodes even if multiple envs
        #             # finished at the same time on the last step.
        #             break

    def reset_envs(self, seed: int | list[int] | None = None) -> None:
        # TODO: Check if the envs were already just reset (with this seed?) and if so, don't reset
        # again.
        # if not any([self.observations, self.actions, self.rewards, self.infos, self.actor_outputs]):
        #     # Envs were already reset?
        #     ...
        logger.info(
            f"Resetting envs. # of wasted env steps: {sum(self.episode_lengths())} ({self.episode_lengths()})"
        )
        if self._episodes_to_yield_at_this_step:
            n_wasted_episodes = len(self._episodes_to_yield_at_this_step)
            n_wasted_steps = sum(ep.length for ep in self._episodes_to_yield_at_this_step)
            logger.warning(
                f"Wasting {n_wasted_episodes} episodes that just completed at this step with "
                f"previous actor but had not yet been yielded (total of {n_wasted_steps} steps)."
            )
            self._episodes_to_yield_at_this_step.clear()

        # Clear the buffers
        self.observations = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.infos = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.actor_outputs = [[] for _ in range(self.num_envs)]

        # Reset all the environments.
        obs_batch, info_batch = self.env.reset(seed=seed)
        self._last_observation = obs_batch
        self._last_info = info_batch
        # Note: not really used atm since we assume the env gives back tensors (not dicts)
        # if isinstance(obs_batch, Mapping):
        #     obs_batch = list(sliced_dict(obs_batch, n_slices=num_envs))
        for i, env_obs in enumerate(obs_batch):
            self.observations[i].append(env_obs)

        env_infos = list(sliced_dict(info_batch, n_slices=self.num_envs))
        for i, env_info in enumerate(env_infos):
            self.infos[i].append(env_info)  # type: ignore

        assert self.episode_lengths() == [1 for _ in range(self.num_envs)]

    @property
    def _should_stop(self) -> bool:
        if self.max_episodes and self._yielded_episodes >= self.max_episodes:
            return True
        if self.max_steps and self._yielded_steps >= self.max_steps:
            return True
        return False

    def episode_lengths(self) -> list[int]:
        """Returns the number of steps in each currently ongoing episode.

        Used mainly for debugging.
        """
        return [len(obs) for obs in self.observations]

    def step_and_yield_completed_episodes(self) -> list[Episode[ActorOutput]]:
        """Do one step in the vectorenv and yield any episodes that just finished at that step."""
        assert self._last_observation is not None, "end should have been reset before stepping!"
        # note: Could pass the info to the actor as well?
        action_batch, actor_output_batch = self.actor(
            self._last_observation, self.env.action_space
        )
        logger.debug(f"{self.episode_lengths()=}")
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = self.env.step(
            action_batch
        )
        self._last_observation = obs_batch
        self._last_info = info_batch
        env_infos = list(sliced_dict(info_batch, n_slices=self.num_envs))
        env_actor_outputs = list(sliced_dict(actor_output_batch, n_slices=self.num_envs))

        episodes_at_this_step: list[Episode[ActorOutput]] = []

        for env_index in range(self.num_envs):
            env_obs = obs_batch[env_index]
            env_reward = reward_batch[env_index]
            env_terminated = terminated_batch[env_index]
            env_truncated = truncated_batch[env_index]
            env_info = env_infos[env_index]
            env_action = action_batch[env_index]
            env_actor_output = env_actor_outputs[env_index]

            if env_terminated | env_truncated:
                # The observation and info are of the first step of the next episode.
                # The action and reward are from the last step of the previous episode.
                self.rewards[env_index].append(env_reward)
                self.actions[env_index].append(env_action)
                self.actor_outputs[env_index].append(env_actor_output)

                # We don't really use these as far as I can tell (the jax envs don't produce them).
                final_observation: Tensor | None = env_info.get("old_observation")
                final_info: EpisodeInfo | None = env_info.get("final_info")

                episode = stack_episode(
                    observations=self.observations[env_index],
                    actions=self.actions[env_index],
                    rewards=self.rewards[env_index],
                    # todo: fix the type here, info is more general than this.
                    infos=self.infos[env_index].copy(),
                    terminated=env_terminated,
                    truncated=env_truncated,
                    actor_outputs=self.actor_outputs[env_index],
                    final_observation=final_observation,
                    final_info=final_info,
                    environment_index=env_index,
                )
                episodes_at_this_step.append(episode)

                self.observations[env_index].clear()
                self.rewards[env_index].clear()
                self.infos[env_index].clear()
                self.actions[env_index].clear()
                self.actor_outputs[env_index].clear()

                # The observation and info are of the first step of the next episode.
                self.observations[env_index].append(env_obs)
                self.infos[env_index].append(env_info)

            else:
                # Ongoing episode. Add the step data to the buffers.
                self.observations[env_index].append(env_obs)
                self.rewards[env_index].append(env_reward)
                self.infos[env_index].append(env_info)
                self.actions[env_index].append(env_action)
                self.actor_outputs[env_index].append(env_actor_output)

        return episodes_at_this_step


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

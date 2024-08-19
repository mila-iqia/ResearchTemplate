from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Generic, TypeVar

import gymnasium
import jax.experimental.compilation_cache.compilation_cache
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from project.datamodules.rl.wrappers.jax_torch_interop import jax_to_torch
from project.utils.types import NestedMapping

from .stacking_utils import stack_episode, unstack
from .types import Actor, ActorOutput, Episode, EpisodeInfo, VectorEnv

logger = get_logger(__name__)
eps = np.finfo(np.float32).eps.item()

jax.experimental.compilation_cache.compilation_cache.set_cache_dir(str(Path.home() / ".cache/jax"))


@dataclass
class EpisodeIterableDataset(IterableDataset[Episode[ActorOutput]], Generic[ActorOutput]):
    """An IterableDataset that uses an actor in an environment and yields episodes."""

    env: gymnasium.Env[Tensor, Tensor] | VectorEnv[Tensor, Tensor]
    actor: Actor[Tensor, Tensor, ActorOutput]
    episodes_per_epoch: int | None = None
    steps_per_epoch: int | None = None
    seed: int | list[int] | None = None
    discount_factor: float | None = None

    def __post_init__(self):
        self.num_envs: int | None = (
            self.env.unwrapped.num_envs
            if isinstance(self.env.unwrapped, gymnasium.vector.VectorEnv)
            else None
        )
        if self.num_envs:
            logger.debug(f"The environment is vectorized (num_envs={self.num_envs})")
        else:
            logger.debug("The environment is NOT vectorized (num_envs is None)")
        # TODO: Need to call a method on the iterator to reset the envs when the actor is updated.
        self._iterator: (
            VectorEnvEpisodeIterator[ActorOutput] | EnvEpisodeIterator[ActorOutput] | None
        ) = None

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
        if isinstance(self._iterator, VectorEnvEpisodeIterator):
            self._iterator.reset_envs()

    def _get_iterator(self):
        if isinstance(self.env.unwrapped, gymnasium.vector.VectorEnv):
            return VectorEnvEpisodeIterator(
                # The env might not be a subclass our `VectorEnv` which just has type hint fixes
                # for `VectorEnv`
                env=self.env,  # type: ignore
                actor=self.actor,
                initial_seed=self.seed,
                max_episodes=self.episodes_per_epoch,
                max_steps=self.steps_per_epoch,
                discount_factor=self.discount_factor,
            )
        else:
            assert self.seed is None or isinstance(self.seed, int)
            return EnvEpisodeIterator(
                env=self.env,
                actor=self.actor,
                initial_seed=self.seed,
                max_episodes=self.episodes_per_epoch,
                max_steps=self.steps_per_epoch,
                discount_factor=self.discount_factor,
            )

    def __iter__(self) -> Iterator[Episode[ActorOutput]]:
        # Reset the env RNG at each epoch start? or only on the first epoch?
        if self._iterator is None:
            self._iterator = self._get_iterator()
        return self._iterator


@dataclasses.dataclass
class EnvEpisodeIterator(Iterator[Episode[ActorOutput]]):
    """Iterator for a single environment that yields episodes one at a time."""

    env: gymnasium.Env[Tensor, Tensor]
    actor: Actor[Tensor, Tensor, ActorOutput]
    max_episodes: int | None = None
    max_steps: int | None = None
    initial_seed: int | None = None

    discount_factor: float | None = None
    """The discount factor (gamma) used to calculate the episode returns at each step."""

    _yielded_steps: int = dataclasses.field(default=0, init=False)
    _yielded_episodes: int = dataclasses.field(default=0, init=False)

    def on_actor_update(self) -> None:
        pass  # do nothing, no buffers to clear or anything like that, really.

    def __next__(self) -> Episode[ActorOutput]:
        if self._should_stop:
            raise StopIteration

        episode = self.collect_one_episode(
            seed=self.initial_seed if self._yielded_episodes == 0 else None
        )
        self._yielded_episodes += 1
        self._yielded_steps += episode.length
        return episode

    def __iter__(self) -> Iterator[Episode[ActorOutput]]:
        return self

    @property
    def _should_stop(self) -> bool:
        if self.max_episodes and self._yielded_episodes >= self.max_episodes:
            return True
        if self.max_steps and self._yielded_steps >= self.max_steps:
            return True
        return False

    def collect_one_episode(self, seed: int | None = None) -> Episode[ActorOutput]:
        # logger.debug(f"Starting episode {self._episode_index}.")
        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards: list[Tensor] = []
        infos: list[dict] = []
        actor_outputs: list[ActorOutput] = []
        terminated = False
        truncated = False
        final_observation: Tensor
        final_info: dict

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
            if isinstance(reward, jax.Array):
                reward = jax_to_torch(reward)
            assert isinstance(reward, torch.Tensor)
            rewards.append(reward)
            if terminated or truncated:
                final_observation = obs
                final_info = info
                break
            else:
                observations.append(obs)
                infos.append(info)

        return stack_episode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            truncated=truncated,
            terminated=terminated,
            actor_outputs=actor_outputs,
            final_observation=final_observation,
            final_info=final_info,
            discount_factor=self.discount_factor,
        )


ActorOutput = TypeVar("ActorOutput", bound=NestedMapping[str, Tensor])


class VectorEnvEpisodeIterator(Iterator[Episode[ActorOutput]]):
    """Yields one episode at a time from interacting with a vectorized environment."""

    def __init__(
        self,
        env: VectorEnv[Tensor, Tensor],
        actor: Actor[Tensor, Tensor, ActorOutput],
        initial_seed: int | list[int] | None = None,
        discount_factor: float | None = None,
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
        self.discount_factor = discount_factor

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
                logger.debug(f"Reached limit of {self.max_episodes} episodes.")
            if self.max_steps and self._yielded_steps >= self.max_steps:
                logger.debug(f"Reached limit of {self.max_steps} steps.")
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

    def reset_envs(self, seed: int | list[int] | None = None) -> None:
        # TODO: Check if the envs were already just reset (with this seed?) and if so, don't reset
        # again.
        # if not any([self.observations, self.actions, self.rewards, self.infos, self.actor_outputs]):
        #     # Envs were already reset?
        #     ...
        logger.debug(
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

        env_infos = list(unstack(info_batch, n_slices=self.num_envs))
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
        env_infos = list(unstack(info_batch, n_slices=self.num_envs))
        env_actor_outputs = list(unstack(actor_output_batch, n_slices=self.num_envs))

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
                final_observation: Tensor | None = env_info.get("final_observation")
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
                    discount_factor=self.discount_factor,
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

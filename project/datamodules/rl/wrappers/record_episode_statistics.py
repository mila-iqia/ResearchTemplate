"""Wrapper that tracks the cumulative rewards and episode lengths."""

import time

import gymnasium.wrappers.record_episode_statistics
import numpy as np


class RecordEpisodeStatistics(
    gymnasium.wrappers.record_episode_statistics.RecordEpisodeStatistics
):
    """Patch for RecordEpisodeStatistics in case of jax Arrays for rewards.

    This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gymnasium.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env, deque_size=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        return super().reset(**kwargs)

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        assert self.episode_returns is not None
        assert self.episode_lengths is not None

        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError("Attempted to add episode stats when they already exist")
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            # These could be jax arrays, so use .where instead of item assignment.
            self.episode_lengths = np.where(
                dones, np.zeros_like(self.episode_lengths), self.episode_lengths
            )
            self.episode_returns = np.where(
                dones, np.zeros_like(self.episode_returns), self.episode_returns
            )
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

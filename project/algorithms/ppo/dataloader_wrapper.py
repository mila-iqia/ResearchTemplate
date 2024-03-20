from __future__ import annotations

import math
from collections.abc import Iterable
from logging import getLogger as get_logger

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm.rich import tqdm_rich

from project.datamodules.rl.rl_dataset import RlDataset

from .utils import (
    PpoEpisodeBatch,
    concatenate_episode_batches,
    dict_slice,
    get_episode_lengths,
)

logger = get_logger(__name__)


class PpoDataLoaderWrapper(Iterable[PpoEpisodeBatch]):
    """Collects a given number of steps, then yields them in minibatches.

    Each "round" begins by collecting at least `min_steps_to_collect_per_round` steps from the
    environment. Then, for `num_epochs_per_round` epochs, the collected data is shuffled and
    yielded in minibatches of `num_episodes_per_batch` episodes each.

    NOTE: The PPO implementations in CleanRL/etc always use a "step-centric" view, while here we
    use "episodes" as the unit. Here we don't shuffle steps within episodes, we keep them whole.
    """

    def __init__(
        self,
        episode_generator: Iterable[PpoEpisodeBatch],
        min_steps_to_collect_per_round: int,
        num_epochs_per_round: int,
        num_episodes_per_batch: int,
        seed: int | None = 42,
    ):
        """
        Parameters
        ----------
        episode_generator: An iterable that yields batches of "live" episodes.
        min_steps_to_collect_per_round: Number of steps to collect at a time
        num_epochs_per_round: Number of times the collected data should be shuffled and yielded \
            before collecting new data.
        num_episodes_per_batch: Number of episodes to yield in each batch (a.k.a. batch size).
        seed: Seed for the shuffling of the collected episodes. Defaults to 42.
        """
        self.episode_generator = episode_generator
        self.min_steps_to_collect_per_round = min_steps_to_collect_per_round
        self.num_epochs_per_round = num_epochs_per_round
        self.num_episodes_per_batch = num_episodes_per_batch
        self.data: PpoEpisodeBatch | None = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._epochs = 0
        self._total_yielded_steps = 0
        self._total_yielded_episodes = 0
        self._round = 0
        self._episode_lengths: list[int] | None = None

        self._max_episode_steps: int | None = None
        if isinstance(self.episode_generator, DataLoader) or hasattr(
            self.episode_generator, "dataset"
        ):
            dataset = getattr(self.episode_generator, "dataset")
            if isinstance(dataset, RlDataset):
                self._max_episode_steps = dataset.env.spec.max_episode_steps

        # Setting these attributes because they can be checked by Pytorch-Lightning to know how to
        # create the progress bar.
        self.dataset = episode_generator
        self.batch_size = num_episodes_per_batch

    def __len__(self) -> int | None:
        if self._max_episode_steps is not None:
            # Adding a +1 here because otherwise the part after the last yield doesn't get run.
            return math.ceil(self.min_steps_to_collect_per_round / self._max_episode_steps) + 1
        if self._episode_lengths:
            return len(self._episode_lengths)
        return None

    def __iter__(self) -> Iterable[PpoEpisodeBatch]:
        epoch_in_round = self._epochs % self.num_epochs_per_round + 1
        dict(
            step=self._total_yielded_steps,
            episode=self._total_yielded_episodes,
            epoch=self._epochs,
            round=self._round,
        )
        logger.debug(
            f"Starting epoch {epoch_in_round}/{self.num_epochs_per_round} of round {self._round}. "
            f"Current step: {self._total_yielded_steps}, episode: {self._total_yielded_episodes}"
        )

        if self.data is None:
            logger.debug(
                f"Collecting at least {self.min_steps_to_collect_per_round} steps in the "
                f"environment."
            )
            self.data, self._episode_lengths = self._collect_data()
            logger.debug(
                f"Done collecting {sum(self._episode_lengths)} steps from "
                f"{len(self._episode_lengths)} episodes in the environment."
            )

        assert self._episode_lengths is not None

        # NOTE: We shuffle the episodes, not the steps within the episodes. This is different than
        # how CleanRL does it.
        n_episodes = len(self._episode_lengths)
        episode_indices = np.arange(n_episodes)

        self.rng.shuffle(episode_indices)
        for start in range(0, n_episodes, self.num_episodes_per_batch):
            episodes_to_yield = episode_indices[start : start + self.num_episodes_per_batch]
            minibatch = dict_slice(self.data, episodes_to_yield)
            _ep_lengths = get_episode_lengths(minibatch)
            self._total_yielded_steps += sum(_ep_lengths)
            self._total_yielded_episodes += len(_ep_lengths)
            yield minibatch

        self._epochs += 1
        if self._epochs % self.num_epochs_per_round == 0:
            logger.info(
                f"End of round {self._round}: Clearing the buffer after {self._epochs} epochs and "
                f"{self._total_yielded_steps} steps."
            )
            self._round += 1
            self.data = None

    @torch.no_grad()
    def _collect_data(self) -> tuple[PpoEpisodeBatch, list[int]]:
        self.data = None

        episode_batches: list[PpoEpisodeBatch] = []
        episode_lengths: list[int] = []
        num_steps = 0
        num_episodes = 0

        console = Console(record=True)
        with tqdm_rich(
            desc="Collecting data from the environment",
            total=self.min_steps_to_collect_per_round,
            unit="Steps",
            options=dict(console=console),
        ) as pbar:
            for new_episodes in self.episode_generator:
                assert num_steps < self.min_steps_to_collect_per_round
                lengths = get_episode_lengths(new_episodes)

                # Add all the episodes.
                episode_batches.append(new_episodes)
                episode_lengths.extend(lengths)
                _total_steps = sum(lengths)
                num_steps += _total_steps
                num_episodes += len(lengths)
                pbar.update(_total_steps)

                # We accumulate more steps than necessary: split the batch and drop the rest.
                # TODO: It might be okay to just have too much data, it's probably not that big a
                # deal.
                # for episode_index, episode_length in enumerate(lengths):
                #     episode = dict_slice(new_episodes, np.array([episode_index], dtype=int))
                #     episode_batches.append(episode)
                #     num_steps += episode_length
                #     pbar.update(episode_length)
                #     num_episodes += 1
                #     n_missing_steps -= episode_length
                #     if n_missing_steps <= 0:
                #         # We've accumulate enough, we can stop here.
                #         break

                if num_steps >= self.min_steps_to_collect_per_round:
                    break

        if num_steps > self.min_steps_to_collect_per_round:
            logger.debug(
                f"Actually collected {num_steps} steps instead of "
                f"{self.min_steps_to_collect_per_round}."
            )

        # Consolidate these loose batches into a single dict of (possibly nested) tensors.
        return concatenate_episode_batches(episode_batches, dim=0), episode_lengths

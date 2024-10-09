"""Utility functions to manage random number generator states."""

import contextlib
import copy
import dataclasses
import random
from contextlib import contextmanager
from typing import Any

import lightning
import numpy as np
import torch


def _get_cuda_rng_states():
    return tuple(
        torch.cuda.get_rng_state(torch.device("cuda", index=index))
        for index in range(torch.cuda.device_count())
    )


@dataclasses.dataclass(frozen=True)
class RngState:
    """Dataclass that contains the state of all the numpy/random/torch RNGs."""

    random_state: tuple[Any, ...] = dataclasses.field(default_factory=random.getstate)
    numpy_random_state: dict[str, Any] = dataclasses.field(default_factory=np.random.get_state)

    torch_cpu_rng_state: torch.Tensor = torch.get_rng_state()
    torch_device_rng_states: tuple[torch.Tensor, ...] = dataclasses.field(
        default_factory=_get_cuda_rng_states
    )

    @classmethod
    def get(cls):
        """Gets the state of the random/numpy/torch random number generators."""
        # Note: do a deepcopy just in case the libraries return the rng state "by reference" and
        # keep modifying it.
        return copy.deepcopy(cls())

    def set(self):
        """Resets the state of the random/numpy/torch random number generators with the contents of
        `self`."""
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.set_rng_state(self.torch_cpu_rng_state)
        for index, state in enumerate(self.torch_device_rng_states):
            torch.cuda.set_rng_state(state, torch.device("cuda", index=index))

    @classmethod
    def seed(cls, base_seed: int):
        lightning.seed_everything(base_seed, workers=True)
        # random.seed(base_seed)
        # np.random.seed(base_seed)
        # torch.random.manual_seed(base_seed)
        return cls()


@contextlib.contextmanager
def fork_rng():
    """Forks the RNG, so that when you return, the RNG is reset to the state that it was previously
    in."""
    # get the global RNG state before
    rng_state = RngState.get()
    # Yield: let the client code modify the global RNG state.
    yield
    # Reset the global RNG state to what it was before.
    rng_state.set()


@contextmanager
def seeded_rng(seed: int = 42):
    """Forks the RNG and seeds the torch, numpy, and random RNGs while inside the block."""
    with fork_rng():
        yield RngState.seed(seed)

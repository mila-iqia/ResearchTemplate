from __future__ import annotations

import contextlib
from collections.abc import Sequence
from logging import getLogger as get_logger
from typing import Any, SupportsFloat

import gym
import gym.spaces
import numpy as np
import torch
from gym import spaces
from gym.core import ActionWrapper, Env, ObservationWrapper
from numpy import ndarray
from torch import LongTensor, Tensor

from .rl_types import TensorType

logger = get_logger(__name__)


class NormalizeBoxActionWrapper(ActionWrapper):
    """Wrapper to normalize gym.spaces.Box actions in [-1, 1].

    TAKEN FROM (https://github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py)
    """

    def __init__(self, env: gym.Env[Any, np.ndarray]):
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError(f"env {env} doesn't have a Box action space.")
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.action_space = type(env.action_space)(
            low=np.ones_like(env.action_space.low) * -1.0,
            high=np.ones_like(env.action_space.high),
            dtype=env.action_space.dtype,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        # rescale the action
        low, high = self.orig_action_space.low, self.orig_action_space.high
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        scaled_action = np.clip(scaled_action, low, high)
        return scaled_action

    def reverse_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.orig_action_space.low, self.orig_action_space.high
        action = (scaled_action - low) * 2.0 / (high - low) - 1.0
        return action


def check_and_normalize_box_actions(env: gym.Env) -> gym.Env:
    """Wrap env to normalize actions if [low, high] != [-1, 1]."""
    if isinstance(env.action_space, spaces.Box):
        low, high = env.action_space.low, env.action_space.high
        if (
            np.abs(low + np.ones_like(low)).max() > 1e-6
            or np.abs(high - np.ones_like(high)).max() > 1e-6
        ):
            logger.info("Normalizing environment actions.")
            return NormalizeBoxActionWrapper(env)

    # Environment does not need to be normalized.
    return env


class ToTensorsWrapper(ObservationWrapper, ActionWrapper, gym.Env[Tensor, Tensor]):
    def __init__(
        self, env: Env[np.ndarray, int] | Env[np.ndarray, np.ndarray], device: torch.device
    ):
        super().__init__(env)
        self.device = device
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = TensorBox(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=env.observation_space.dtype,
            device=device,
            # seed=env.seed,
        )
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = TensorBox(
                low=env.action_space.low,
                high=env.action_space.high,
                dtype=env.action_space.dtype,
                device=device,
                # seed=env.seed,
            )
        else:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.action_space = TensorDiscrete(
                n=env.action_space.n,
                start=env.action_space.start,
                # seed=env.seed,
            )

    def reset(
        self, seed: int | None = None, options: dict | None = None, **kwargs
    ) -> tuple[Tensor, dict]:
        """Resets the environment, returning a modified observation using
        :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self.observation(obs), info

    def step(self, action: Tensor) -> tuple[Tensor, float, bool, bool, dict]:
        """Returns a modified observation using :meth:`self.observation` after calling
        :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: np.ndarray) -> Tensor:
        return torch.as_tensor(
            observation, dtype=self.observation_space.torch_dtype, device=self.device
        )

    def action(self, action: Tensor) -> np.ndarray:
        return action.detach().cpu().numpy()


class TensorDiscrete(gym.spaces.Discrete, gym.spaces.Space[LongTensor]):
    def __init__(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        start: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(n, seed, start)
        self.device = device

    def sample(self, mask: ndarray | None = None) -> Tensor:
        return torch.as_tensor(super().sample(mask), device=self.device)

    def contains(self, x) -> bool:
        if isinstance(x, Tensor):
            x = x.item()
        return super().contains(x)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.start != 0:
            return f"{class_name}({self.n}, start={self.start}, device={self.device})"
        return f"{class_name}({self.n}, device={self.device})"


class TensorBox(gym.spaces.Box, gym.spaces.Space[TensorType]):
    def __init__(
        self,
        low: SupportsFloat | ndarray,
        high: SupportsFloat | np.ndarray,
        shape: Sequence[int] | None = None,
        dtype: np.dtype | type = np.float32,
        seed: int | np.random.Generator | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.np_dtype, self.torch_dtype = get_numpy_and_torch_dtypes(dtype)
        super().__init__(low=low, high=high, shape=shape, dtype=self.np_dtype, seed=seed)
        self.device = device
        assert isinstance(self.torch_dtype, torch.dtype)
        self.low_tensor = torch.as_tensor(self.low, dtype=self.torch_dtype, device=self.device)
        self.high_tensor = torch.as_tensor(self.high, dtype=self.torch_dtype, device=self.device)
        # todo: Can we let the dtype be a torch dtype instead?
        self.dtype = self.torch_dtype

    @contextlib.contextmanager
    def _use_np_dtype(self):
        dtype_before = self.dtype
        self.dtype = self.np_dtype
        yield
        self.dtype = dtype_before

    def sample(self, mask: None = None) -> Tensor:
        with self._use_np_dtype():
            return torch.as_tensor(
                super().sample(mask), dtype=self.torch_dtype, device=self.device
            )

    def contains(self, x) -> bool:
        if isinstance(x, Tensor):
            return (
                torch.can_cast(x.dtype, self.torch_dtype)
                and (x.shape == self.shape)
                and bool(torch.all(x >= self.low_tensor).item())
                and bool(torch.all(x <= self.high_tensor).item())
            )
        return super().contains(x)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype}, "
            f"device={self.device})"
        )

    # def __repr__(self) -> str:
    #     class_name = type(self).__name__
    #     if self.start != 0:
    #         return f"{class_name}({self.n}, start={self.start}, device={self.device})"
    #     return f"{class_name}({self.n}, device={self.device})"


def get_numpy_and_torch_dtypes(dtype: np.dtype | type) -> tuple[np.dtype, torch.dtype]:
    # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
    numpy_to_torch_dtype = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }
    torch_to_numpy_dtype = {v: k for k, v in numpy_to_torch_dtype.items()}
    # TODO: DEBUG later. == works but `dtype in <dict>` doesn't.
    if dtype == np.float32:
        return dtype, torch.float32
    if dtype in numpy_to_torch_dtype:
        return dtype, numpy_to_torch_dtype[dtype]
    if dtype in torch_to_numpy_dtype:
        return torch_to_numpy_dtype[dtype], dtype
    raise ValueError(f"Invalid dtype {dtype} (type {type(dtype)})")

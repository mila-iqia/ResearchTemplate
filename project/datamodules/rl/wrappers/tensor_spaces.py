import contextlib
from collections.abc import Sequence
from typing import SupportsFloat

import gym.spaces
import numpy as np
import torch
from numpy import ndarray
from torch import LongTensor, Tensor

from ..rl_types import TensorType


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
        self.torch_dtype: torch.dtype
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


class TensorDiscrete(gym.spaces.Discrete, gym.spaces.Space[LongTensor]):
    def __init__(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        start: int = 0,
        dtype: np.dtype | type = np.int32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(n, seed, start)
        self.device = device
        self.np_dtype, self.torch_dtype = get_numpy_and_torch_dtypes(dtype)

    def sample(self, mask: ndarray | None = None) -> Tensor:
        return torch.as_tensor(super().sample(mask), device=self.device, dtype=self.torch_dtype)

    def contains(self, x) -> bool:
        if isinstance(x, Tensor):
            x = x.item()
        return super().contains(x)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.start != 0:
            return f"{class_name}({self.n}, start={self.start}, device={self.device})"
        return f"{class_name}({self.n}, device={self.device})"

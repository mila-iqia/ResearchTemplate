from collections.abc import Sequence
from logging import getLogger as get_logger
from typing import Any, SupportsFloat

import gymnasium
import gymnasium.spaces
import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpy.typing import NDArray
from torch import LongTensor, Tensor
from typing_extensions import TypeVar

from project.utils.device import default_device

from ..rl_types import TensorType

logger = get_logger(__name__)


def get_jax_dtype(torch_dtype: torch.dtype) -> jax.numpy.dtype:
    return {
        torch.bool: jnp.bool_,
        torch.uint8: jnp.uint8,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.complex64: jnp.complex64,
        torch.complex128: jnp.complex128,
    }[torch_dtype]


ArrayType = TypeVar("ArrayType", bound=NDArray, default=NDArray)


# todo: make this not a subclass of gymnasium.spaces.Box and instead make it "have a" instead of
# "be a" Box. This would probably require us to register a few of the handlers for the Box spaces
# to be used for TensorBox spaces in things like `gymnasium.vector.utils.batch_space` and such.
class TensorBox(gymnasium.spaces.Space[TensorType]):
    def __init__(
        self,
        low: SupportsFloat | np.ndarray | Tensor,
        high: SupportsFloat | np.ndarray | Tensor,
        shape: Sequence[int] | None = None,
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        device: torch.device = default_device(),
    ):
        if not dtype:
            if isinstance(low, Tensor):
                dtype = low.dtype
            elif isinstance(high, Tensor):
                dtype = high.dtype
            else:
                dtype = torch.float32

        if not shape:
            if isinstance(low, Tensor | np.ndarray):
                shape = low.shape
            elif isinstance(high, Tensor | np.ndarray):
                shape = high.shape
            else:
                shape = ()

        # needs to be set for super init to call self.seed
        self.device = device
        self._rng = torch.Generator(device=self.device)
        # not passing `dtype` because it assumes it's a numpy dtype.
        super().__init__(shape=shape, dtype=None, seed=seed)
        self.dtype = dtype
        self.low = torch.as_tensor(low, dtype=self.dtype, device=self.device)
        self.high = torch.as_tensor(high, dtype=self.dtype, device=self.device)
        if self.shape and self.low.shape != self.shape:
            self.low = self.low.expand(self.shape)
        if self.shape and self.high.shape != self.shape:
            self.high = self.high.expand(self.shape)
        assert self.low.shape == shape
        assert self.high.shape == shape

    def seed(self, seed: int) -> None:
        self._rng.manual_seed(seed)

    def sample(self) -> Tensor:
        """Generates a single random sample inside the Box."""
        rand = torch.rand(
            size=self.low.shape, dtype=self.dtype, device=self.device, generator=self._rng
        )
        return self.low + rand * (self.high - self.low)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, Tensor)
            and torch.can_cast(x.dtype, self.dtype)
            and (x.shape == self.shape)
            and bool(
                ((x_sametype := x.type_as(self.low)) >= self.low).all()
                & (x_sametype <= self.high).all()
            )
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}({self.low}, {self.high}, {self.shape}, {self.dtype}, "
            f"device={self.device})"
        )


class TensorDiscrete(gymnasium.spaces.Space[LongTensor]):
    def __init__(
        self,
        n: int,
        start: int = 0,
        dtype: torch.dtype = torch.int32,
        seed: int | None = None,
        device: torch.device = default_device(),
    ):
        self.n = n
        self.start = start
        self.device = device
        self.dtype = dtype
        self._rng = torch.Generator(device=self.device)
        super().__init__(shape=(), dtype=None, seed=seed)
        self.shape: tuple[int, ...]

    def seed(self, seed: int):
        self._rng.manual_seed(seed)

    def sample(self) -> Tensor:
        return torch.randint(
            low=self.start,
            high=self.n,
            size=self.shape,
            dtype=self.dtype,
            device=self.device,
            generator=self._rng,
        )

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, Tensor)
            and torch.can_cast(x.dtype, self.dtype)
            and (x.shape == self.shape)
            and bool((x >= self.start).all() & (x <= self.n).all())
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.start != 0:
            return f"{class_name}({self.n}, start={self.start}, device={self.device})"
        return f"{class_name}({self.n}, device={self.device})"


def get_torch_dtype_from_numpy_dtype(dtype: np.dtype) -> torch.dtype:
    if dtype == np.float32:
        # note: getitem doesn't work for np.float32?
        return torch.float32
    return {
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
    }[dtype]


# def get_numpy_and_torch_dtypes(dtype: np.dtype | type) -> tuple[np.dtype, torch.dtype]:
#     # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
#     numpy_to_torch_dtype = {
#         np.bool_: torch.bool,
#         np.uint8: torch.uint8,
#         np.int8: torch.int8,
#         np.int16: torch.int16,
#         np.int32: torch.int32,
#         np.int64: torch.int64,
#         np.float16: torch.float16,
#         np.float32: torch.float32,
#         np.float64: torch.float64,
#         np.complex64: torch.complex64,
#         np.complex128: torch.complex128,
#     }
#     torch_to_numpy_dtype = {v: k for k, v in numpy_to_torch_dtype.items()}
#     # TODO: DEBUG later. == works but `dtype in <dict>` doesn't.
#     if dtype == np.float32:
#         return dtype, torch.float32
#     if dtype in numpy_to_torch_dtype:
#         return dtype, numpy_to_torch_dtype[dtype]
#     if dtype in torch_to_numpy_dtype:
#         return torch_to_numpy_dtype[dtype], dtype
#     raise ValueError(f"Invalid dtype {dtype} (type {type(dtype)})")

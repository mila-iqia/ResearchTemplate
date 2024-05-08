from collections.abc import Sequence
from logging import getLogger as get_logger
from types import ModuleType
from typing import Any, SupportsFloat

import chex
import gymnasium
import gymnasium.spaces
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing
import torch
from torch import Tensor

from project.datamodules.rl.wrappers.jax_torch_interop import (
    jax_to_torch_tensor,
    torch_to_jax_tensor,
)

logger = get_logger(__name__)


class TensorSpace(gymnasium.spaces.Space[torch.Tensor]):
    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: torch.dtype | None = None,
        seed: int | torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self._rng: torch.Generator = torch.Generator(self.device)
        # Note: passing `dtype=None` because super init doesn't handle torch dtypes (only numpy).
        super().__init__(shape, dtype=None, seed=None)
        self.dtype: torch.dtype | None = dtype
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int | torch.Tensor) -> None:
        if isinstance(seed, int):
            self._rng = self._rng.manual_seed(seed)
        else:
            self._rng = self._rng.set_state(seed)


# TODO: Make a PR to add this unbounded sampling to the gymnax Box space.
class TensorBox(TensorSpace):
    def __init__(
        self,
        low: SupportsFloat | np.ndarray | Tensor,
        high: SupportsFloat | np.ndarray | Tensor,
        shape: Sequence[int] | None = None,
        dtype: torch.dtype | None = None,
        seed: int | torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
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
        # not passing `dtype` because it assumes it's a numpy dtype.
        self._key: jax.Array = jax.random.key(0)
        super().__init__(shape=shape, dtype=dtype, seed=seed, device=device)
        self.dtype: torch.dtype
        self.low = torch.as_tensor(low, dtype=self.dtype, device=self.device)
        self.high = torch.as_tensor(high, dtype=self.dtype, device=self.device)

        if self.shape and self.low.shape != self.shape:
            self.low = self.low.expand(self.shape)
        if self.shape and self.high.shape != self.shape:
            self.high = self.high.expand(self.shape)
        assert self.low.shape == shape
        assert self.high.shape == shape
        self._jax_high = torch_to_jax_tensor(self.high)
        self._jax_low = torch_to_jax_tensor(self.low)
        assert not jax.numpy.isnan(self._jax_low).any()
        assert not jax.numpy.isnan(self._jax_high).any()
        self.bounded_below: jax.Array = self._jax_low > jax.numpy.finfo(self._jax_low.dtype).min
        self.bounded_above: jax.Array = self._jax_high < jax.numpy.finfo(self._jax_low.dtype).max
        self._jax_dtype = self._jax_low.dtype

    # def to(self, device: torch.device) -> Self:
    #     return type(self)(
    #         low=self.low.to(device=device),
    #         high=self.high.to(device=device),
    #         shape=self.shape,
    #         dtype=self.dtype,
    #         seed=None,
    #         device=device,
    #     )

    # def cuda(self, index: int | None = None) -> Self:
    #     if index:
    #         device = torch.device("cuda", index=index)
    #     else:
    #         device = torch.device("cuda")
    #     return self.to(device=device)

    # def cpu(self) -> Self:
    #     return self.to(device=torch.device("cpu"))

    def seed(self, seed: int) -> None:
        self._rng.manual_seed(seed)
        self._key = jax.random.key(seed)

    def sample(self) -> Tensor:
        """Generates a single random sample inside the Box."""
        # rand = torch.rand(
        #     size=self.low.shape, dtype=self.dtype, device=self.device, generator=self._rng
        # )
        # TODO: Support unbounded intervals like gymnasium.spaces.Box.
        # if self.low.isfinite().all() and self.high.isfinite().all():
        #     return self.low + rand * (self.high - self.low)
        self._key, jax_sample = self._sample(key=self._key)
        return jax_to_torch_tensor(jax_sample)

    def _sample(self, key: jax.Array) -> tuple[chex.PRNGKey, jax.Array]:
        r"""In creating a sample of the box, each coordinate is sampled (independently) from a
        distribution that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution


        Returns:
            the new random key and a sampled value from the Box
        """

        # Adapted from gymnasium.spaces.Box.sample
        high = (
            self._jax_high
            if self._jax_high.dtype.kind == "f"
            else self._jax_high.astype("int64") + 1
        )
        sample = jnp.zeros_like(self._jax_low)
        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above
        new_key, unbounded_key, low_bounded_key, upp_bounded_key, bounded_key = jax.random.split(
            key, 5
        )
        # Vectorized sampling by interval type
        sample = sample.at[unbounded].set(
            jax.random.normal(
                unbounded_key, shape=unbounded[unbounded].shape, dtype=self._jax_dtype
            )
        )

        sample = sample.at[low_bounded].set(
            jax.random.exponential(
                low_bounded_key, shape=low_bounded[low_bounded].shape, dtype=self._jax_dtype
            )
            + self._jax_low[low_bounded]
        )

        sample = sample.at[upp_bounded].set(
            -jax.random.exponential(
                upp_bounded_key, shape=upp_bounded[upp_bounded].shape, dtype=self._jax_dtype
            )
            + high[upp_bounded]
        )

        sample = sample.at[bounded].set(
            jax.random.uniform(
                bounded_key,
                minval=self._jax_low[bounded],
                maxval=high[bounded],
                shape=bounded[bounded].shape,
                dtype=self._jax_dtype,
            )
        )

        if self._jax_dtype.kind in ["i", "u", "b"]:
            sample = jnp.floor(sample)

        return new_key, sample.astype(self._jax_dtype)

    def contains(self, x: Any) -> bool:
        # BUG: doesn't work with `nan` values for low or high.
        min_value = torch.finfo(self.dtype).min
        max_value = torch.finfo(self.dtype).max
        return (
            isinstance(x, Tensor)
            and torch.can_cast(x.dtype, self.dtype)
            and (x.shape == self.shape)
            and (x.device == self.device)  # avoid unintentionally moving things between devices.
            and not bool(x.isnan().any())
            and bool((x >= torch.nan_to_num(self.low, nan=min_value, neginf=min_value)).all())
            and bool((x <= torch.nan_to_num(self.high, nan=max_value, posinf=max_value)).all())
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}({self.low}, {self.high}, {self.shape}, {self.dtype}, "
            f"device={self.device})"
        )


class TensorDiscrete(TensorSpace):
    def __init__(
        self,
        n: int,
        start: int = 0,
        dtype: torch.dtype = torch.int32,
        seed: int | torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(shape=(), dtype=dtype, seed=seed, device=device)
        self.n = n
        self.start = start
        self.shape: tuple[int, ...]
        self.dtype: torch.dtype

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
            and (x.device == self.device)  # avoid unintentionally moving things between devices.
            and bool((x >= self.start).all() & (x < (self.start + self.n)).all())
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.start != 0:
            return f"{class_name}({self.n}, start={self.start}, device={self.device})"
        return f"{class_name}({self.n}, device={self.device})"


@gymnasium.vector.utils.batch_space.register(TensorBox)
def _batch_tensor_box_space(space: TensorBox, n: int = 1) -> TensorBox:
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = torch.tile(space.low, repeats), torch.tile(space.high, repeats)
    rng_state = space._rng.get_state()
    space = type(space)(low=low, high=high, dtype=space.dtype, seed=None, device=space.device)
    space._rng.set_state(rng_state)
    return space


class TensorMultiDiscrete(TensorSpace):
    """Adapted from gymnasium.spaces.MultiDiscrete.

    TODO: Maybe make a PR to gymnax to add this space (with only jax) if they want it?
    """

    def __init__(
        self,
        nvec: torch.Tensor | numpy.typing.NDArray[np.integer[Any]] | list[int],
        dtype: torch.dtype = torch.int64,
        seed: int | torch.Tensor | None = None,
        start: torch.Tensor | numpy.typing.NDArray[np.integer[Any]] | list[int] | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Constructor of :class:`MultiDiscrete` space. (adapted from )

        The argument ``nvec`` will determine the number of values each categorical variable can take. If
        ``start`` is provided, it will define the minimal values corresponding to each categorical variable.

        Args:
            nvec: vector of counts of each categorical variable. This will usually be a list of integers. However,
                you may also pass a more complicated numpy array if you'd like the space to have several axes.
            dtype: This should be some kind of integer type.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            start: Optionally, the starting value the element of each class will take (defaults to 0).
        """
        self.device = device
        self.dtype = dtype
        if isinstance(nvec, torch.Tensor):
            assert nvec.device == device  # just to avoid any confusion.
        self.nvec = torch.as_tensor(nvec, dtype=dtype, device=device)
        if start is not None:
            self.start = torch.as_tensor(start, dtype=dtype, device=device)
        else:
            self.start = torch.zeros(self.nvec.shape, dtype=dtype, device=device)

        assert (
            self.start.shape == self.nvec.shape
        ), "start and nvec (counts) should have the same shape"
        assert (self.nvec > 0).all(), "nvec (counts) have to be positive"

        super().__init__(self.nvec.shape, dtype, seed=seed, device=device)
        self.shape: tuple[int, ...]

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: tuple | None = None) -> torch.Tensor:
        """Generates a single random sample this space.

        Args:
            mask: An optional mask for multi-discrete, expects tuples with a `np.ndarray` mask in the position of each
                action with shape `(n,)` where `n` is the number of actions and `dtype=np.int8`.
                Only mask values == 1 are possible to sample unless all mask values for an action are 0 then the default action `self.start` (the smallest element) is sampled.

        Returns:
            An `np.ndarray` of shape `space.shape`
        """
        if mask is not None:
            raise NotImplementedError("TODO: Implement masking.")

        return (
            torch.rand(size=self.nvec.shape, generator=self._rng, device=self.device) * self.nvec
        ).to(dtype=self.dtype) + self.start

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        if not isinstance(x, torch.Tensor):
            return False
        if x.device != self.device:
            logger.warning(
                f"{type(self).__name__}.contains: Refusing to compare tensors across devices ({x.device=}, {self.device=}). returning False."
            )
            return False

        return bool(
            x.shape == self.shape
            and x.dtype != object
            and (self.start <= x).all()
            and (x - self.start < self.nvec).all()
        )


@gymnasium.vector.utils.batch_space.register(TensorDiscrete)
def _batch_tensor_discrete_space(space: TensorDiscrete, n: int = 1) -> TensorMultiDiscrete:
    # TODO: would need to implement something like MultiDiscrete? or what?
    return TensorMultiDiscrete(
        nvec=torch.full((n,), space.n, dtype=space.dtype, device=space.device),
        dtype=space.dtype,
        seed=space._rng.get_state(),
        start=torch.full((n,), space.start, dtype=space.dtype, device=space.device),
        device=space.device,
    )


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


def get_torch_dtype(dtype: np.dtype | jnp.dtype, np: ModuleType = np) -> torch.dtype:
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


def get_torch_dtype_from_jax_dtype(dtype: jnp.dtype) -> torch.dtype:
    return get_torch_dtype(dtype, np=jnp)


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

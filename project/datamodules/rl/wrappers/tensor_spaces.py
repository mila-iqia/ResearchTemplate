from collections.abc import Sequence
from logging import getLogger as get_logger
from types import ModuleType
from typing import Any, SupportsFloat

import chex
import gymnasium
import gymnasium.spaces
import gymnasium.spaces.utils
import jax
import jax.numpy as jnp
import numpy as np
import numpy.core
import numpy.core.numeric
import numpy.core.numerictypes
import numpy.typing
import torch
from torch import Tensor

from project.datamodules.rl.wrappers.jax_torch_interop import (
    chexify,
    jax_to_torch_tensor,
    jit,
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
    """Tensor version of gymnasium.spaces.Box.

    Samples in the interval [low, high] (inclusive).
    """

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
        self.shape: tuple[int, ...]
        self.low = torch.as_tensor(low, dtype=self.dtype, device=self.device)
        self.high = torch.as_tensor(high, dtype=self.dtype, device=self.device)
        if self.dtype.is_floating_point:
            min_value = torch.finfo(self.dtype).min
            max_value = torch.finfo(self.dtype).max
        else:
            min_value = torch.iinfo(self.dtype).min
            max_value = torch.iinfo(self.dtype).max
        if self.shape and self.low.shape != self.shape:
            self.low = self.low.expand(self.shape)
        if self.shape and self.high.shape != self.shape:
            self.high = self.high.expand(self.shape)
        self.low = torch.nan_to_num(self.low, nan=min_value, neginf=min_value)
        self.high = torch.nan_to_num(self.high, nan=max_value, posinf=max_value)

        assert self.low.shape == shape
        assert self.high.shape == shape
        self._jax_high = torch_to_jax_tensor(self.high.contiguous())
        self._jax_low = torch_to_jax_tensor(self.low.contiguous())
        self._jax_dtype = self._jax_low.dtype

    def seed(self, seed: int | torch.Tensor) -> None:
        super().seed(seed)
        if not isinstance(seed, int):
            # Make an int based on this `seed` tensor.
            seed = int(torch.randint(0, 2**32, generator=self._rng).item())
        self._key = jax.random.key(seed)

    def sample(self) -> Tensor:
        """Generates a single random sample inside the Box."""
        # Use the jit-ed function to sample from the box.
        self._key, jax_sample = box_sample(
            key=self._key,
            low=self._jax_low,
            high=self._jax_high,
        )
        chex.block_until_chexify_assertions_complete()
        torch_tensor = jax_to_torch_tensor(jax_sample)
        # Seems like the sample can be of dtype torch.float32 even if ours is torch.float64
        return torch_tensor.to(dtype=self.dtype)

    def contains(self, x: Any) -> bool:
        # BUG: doesn't work with `nan` values for low or high.
        return bool(
            isinstance(x, Tensor)
            and torch.can_cast(x.dtype, self.dtype)
            and (x.shape == self.shape)
            and (x.device == self.device)  # avoid unintentionally moving things between devices.
            and not bool(x.isnan().any())
            and (x >= self.low).all()
            and (x <= self.high).all()
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return (
            f"{class_name}(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    def __eq__(self, other: Any) -> bool:
        return bool(
            isinstance(other, type(self))
            and other.dtype == self.dtype
            and other.device == self.device
            and other.low.equal(self.low)
            and other.high.equal(self.high)
        )


@chexify
@jit
def box_sample(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
) -> tuple[chex.PRNGKey, jax.Array]:
    r"""Adapted from gymnasium.spaces.Box.sample.

    In creating a sample of the box, each coordinate is sampled (independently) from a
    distribution that is chosen according to the form of the interval:

    * :math:`[a, b]` : uniform distribution
    * :math:`[a, \infty)` : shifted exponential distribution
    * :math:`(-\infty, b]` : shifted negative exponential distribution
    * :math:`(-\infty, \infty)` : normal distribution


    Returns:
        the new random key and a sampled value from the Box
    """
    chex.assert_equal_shape([low, high])
    chex.assert_trees_all_equal_dtypes(low, high)
    dtype = low.dtype

    if dtype.kind in ["f", "c"]:
        bounded_below: jax.Array = low > jax.numpy.finfo(dtype).min
        bounded_above: jax.Array = high < jax.numpy.finfo(dtype).max
    else:
        bounded_below: jax.Array = low > jax.numpy.iinfo(dtype).min
        bounded_above: jax.Array = high < jax.numpy.iinfo(dtype).max

    high = high if high.dtype.kind == "f" else high.astype("int64") + 1

    # Masking arrays which classify the coordinates according to interval type
    # unbounded = ~bounded_below & ~bounded_above  # note: not used, see code below.
    upp_bounded = ~bounded_below & bounded_above
    low_bounded = bounded_below & ~bounded_above
    bounded = bounded_below & bounded_above

    new_key, normal_key, exponential_key, uniform_key = jax.random.split(key, 4)

    # Vectorized sampling by interval type
    # Seems like we need the shapes to be static for jit to work, so we sample more values than we
    # need and then slice them up.

    # NOTE: We can sample the exponential only once and use it twice without fear of having the
    # same values because the masks don't overlap.
    # todo: this needs to be a float dtype.
    float_dtype = dtype if dtype.kind in ["f", "c"] else jnp.float32
    normal_var = jax.random.normal(normal_key, shape=low.shape, dtype=float_dtype)
    # values don't matter in the False case, so I'm adding a dummy 0 here just to avoid any
    # potential issues with nans in the sampling.
    exponential_var = jax.random.exponential(exponential_key, shape=low.shape, dtype=float_dtype)
    uniform_var = jax.random.uniform(
        uniform_key,
        minval=jnp.where(bounded, low, jnp.zeros_like(low)),
        maxval=jnp.where(bounded, high, jnp.ones_like(low)),
        shape=low.shape,
        dtype=float_dtype,
    )
    # note: Since the `sample` gets all filled, it could actually be one of the random variables,
    # it wouldn't change anything, so we could actually save one step!
    # sample = jnp.zeros_like(low)
    # sample = jnp.where(unbounded, normal_var, sample)
    sample = normal_var
    sample = jnp.where(low_bounded, low + exponential_var, sample)
    sample = jnp.where(upp_bounded, high - exponential_var, sample)
    sample = jnp.where(bounded, uniform_var, sample)
    chex.assert_tree_all_finite(sample)
    if dtype.kind in ["i", "u", "b"]:
        sample = jnp.floor(sample)

    return new_key, sample.astype(low.dtype)


gymnasium.spaces.utils.flatdim.register(
    TensorBox, gymnasium.spaces.utils.flatdim.dispatch(gymnasium.spaces.Box)
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
        return self.start + torch.randint(
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
            return f"{class_name}({self.n}, start={self.start}, dtype={self.dtype}, device={self.device})"
        return f"{class_name}({self.n}, device={self.device})"

    def __eq__(self, other: Any) -> bool:
        return bool(
            isinstance(other, type(self))
            and other.dtype == self.dtype
            and other.device == self.device
            and other.start == self.start
            and other.n == self.n
        )


# Reuse the flatdim implementation for Discrete spaces.
gymnasium.spaces.utils.flatdim.register(
    TensorDiscrete, gymnasium.spaces.utils.flatdim.dispatch(gymnasium.spaces.Discrete)
)


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

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if (self.start != 0).any():
            return f"{class_name}(start={self.start}, nvec={self.nvec}, dtype={self.dtype}, device={self.device})"
        return f"{class_name}({self.nvec}, dtype={self.dtype}, device={self.device})"

    def __eq__(self, other: Any) -> bool:
        return bool(
            isinstance(other, type(self))
            and other.dtype == self.dtype
            and other.device == self.device
            and other.start.equal(self.start)
            and other.nvec.equal(self.nvec)
        )


@gymnasium.spaces.utils.flatdim.register(TensorMultiDiscrete)
def _flatdim_multidiscrete(space: TensorMultiDiscrete) -> int:
    return int(torch.sum(space.nvec).item())


@gymnasium.vector.utils.batch_space.register(TensorDiscrete)
def _batch_tensor_discrete_space(space: TensorDiscrete, n: int = 1) -> TensorMultiDiscrete:
    # Based on this from MultiDiscrete:
    # MultiDiscrete(
    #     np.full((n,), space.n, dtype=space.dtype),
    #     dtype=space.dtype,
    #     seed=deepcopy(space.np_random),
    #     start=np.full((n,), space.start, dtype=space.dtype),
    # )
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

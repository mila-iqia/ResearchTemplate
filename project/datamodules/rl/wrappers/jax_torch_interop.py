import collections.abc
import functools
from typing import Any

import jax
import torch
from jax import dlpack as jax_dlpack
from torch import Tensor
from torch.utils import dlpack as torch_dlpack


@functools.singledispatch
def torch_to_jax(value: Any) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Args:
      value: torch tensor

    Returns:
      a JAX array
    """
    raise NotImplementedError(
        f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    )


def torch_tensor_to_jax_array(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array."""
    tensor = torch_dlpack.to_dlpack(value)  # type: ignore
    tensor = jax_dlpack.from_dlpack(tensor)  # type: ignore
    return tensor


# Register it like this so the type hints are preserved on `torch_tensor_to_jax_array` (which is
# also called directly in some places).
torch_to_jax.register(torch.Tensor, torch_tensor_to_jax_array)


@torch_to_jax.register(collections.abc.Mapping)
def _torch_dict_to_jax(value: dict[str, torch.Tensor | Any]) -> dict[str, jax.Array | Any]:
    """Converts a dict of PyTorch tensors into a dict of jax.Arrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


@functools.singledispatch
def jax_to_torch(value: Any) -> Any:
    raise NotImplementedError(
        f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    )


@jax_to_torch.register(type(None))
@torch_to_jax.register(type(None))
def _no_op(v: Any) -> Any:
    return v


def jax_array_to_torch_tensor(value: jax.Array) -> Tensor:
    dpack = jax_dlpack.to_dlpack(value)  # type: ignore
    return torch_dlpack.from_dlpack(dpack)


jax_to_torch.register(jax.Array, jax_array_to_torch_tensor)


@jax_to_torch.register(collections.abc.Mapping)
def _jax_dict_to_torch(value: dict[str, jax.Array | Any]) -> dict[str, torch.Tensor | Any]:
    """Converts a dict of Jax arrays into a dict of PyTorch tensors ."""
    return type(value)(**{k: jax_to_torch(v) for k, v in value.items()})  # type: ignore

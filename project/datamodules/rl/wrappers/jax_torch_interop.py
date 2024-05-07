import collections.abc
import dataclasses
import functools
from typing import Any

import jax
import torch
from jax import dlpack as jax_dlpack
from torch import Tensor
from torch.utils import dlpack as torch_dlpack

from project.utils.types import NestedDict, NestedMapping
from project.utils.types.protocols import Dataclass


@functools.singledispatch
def torch_to_jax(value: Any, /) -> Any:
    """Converts PyTorch tensors to JAX arrays.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: torch tensor

    Returns:
      a JAX array
    """
    raise NotImplementedError(
        f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    )


@functools.singledispatch
def jax_to_torch(value: Any, /) -> Any:
    """Converts JAX arrays to PyTorch Tensors.

    Converts the tensors "in-place", without the need for copies or moving data to the CPU.

    Args:
      value: jax array

    Returns:
      a PyTorch tensor
    """
    raise NotImplementedError(
        f"No registered handler for converting value of type {type(value)} to jax! (value={value})"
    )


def torch_to_jax_tensor(value: torch.Tensor, /) -> jax.Array:
    """Converts a PyTorch Tensor into a jax.Array."""
    tensor = torch_dlpack.to_dlpack(value)  # type: ignore
    tensor = jax_dlpack.from_dlpack(tensor)  # type: ignore
    return tensor


def jax_to_torch_tensor(value: jax.Array, /) -> Tensor:
    dpack = jax_dlpack.to_dlpack(value)  # type: ignore
    return torch_dlpack.from_dlpack(dpack)


# Register it like this so the type hints are preserved on `torch_tensor_to_jax_array` (which is
# also called directly in some places).
torch_to_jax.register(torch.Tensor, torch_to_jax_tensor)
jax_to_torch.register(jax.Array, jax_to_torch_tensor)


@torch_to_jax.register(collections.abc.Mapping)
def torch_to_jax_dict[K](value: NestedMapping[K, Tensor], /) -> NestedMapping[K, jax.Array]:
    """Converts a dict of PyTorch tensors into a dict of jax.Arrays."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


# Keep `None`s the same.
@jax_to_torch.register(type(None))
@torch_to_jax.register(type(None))
def _no_op(v: Any, /) -> Any:
    return v


@jax_to_torch.register(collections.abc.Mapping)
def _jax_dict_to_torch(value: dict[str, jax.Array | Any], /) -> dict[str, torch.Tensor | Any]:
    """Converts a dict of Jax arrays into a dict of PyTorch tensors ."""
    return type(value)(**{k: jax_to_torch(v) for k, v in value.items()})  # type: ignore


class _DataclassMeta(type):
    def __subclasscheck__(self, subclass: type) -> bool:
        return dataclasses.is_dataclass(subclass) and not dataclasses.is_dataclass(type(subclass))

    def __instancecheck__(self, instance: Any) -> bool:
        return dataclasses.is_dataclass(instance) and dataclasses.is_dataclass(type(instance))


# NOTE: Not using a `runtime_checkable` version of the `Dataclass` protocol here, because it
# doesn't work correctly in the case of `isinstance(SomeDataclassType, Dataclass)`, which returns
# `True` when it should be `False` (since it's a dataclass type, not a dataclass instance), and the
# runtime_checkable decorator doesn't check the type of the attribute (ClassVar vs instance
# attribute).
class _DataclassInstance(metaclass=_DataclassMeta): ...


@jax_to_torch.register(_DataclassInstance)
def jax_dataclass_to_torch_dataclass(value: Dataclass, /) -> NestedDict[str, torch.Tensor]:
    return jax_to_torch(dataclasses.asdict(value))


@torch_to_jax.register(_DataclassInstance)
def torch_dataclass_to_jax_dataclass(value: Dataclass, /) -> NestedDict[str, jax.Array]:
    return torch_to_jax(dataclasses.asdict(value))

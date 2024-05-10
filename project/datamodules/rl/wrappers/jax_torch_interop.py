import collections.abc
import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Concatenate

import chex
import gymnasium
import jax
import torch
from jax import dlpack as jax_dlpack
from torch import Tensor
from torch.utils import dlpack as torch_dlpack

from project.datamodules.rl.rl_types import VectorEnv
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
    """Converts a PyTorch Tensor into a jax.Array.

    NOTE: seems like torch.float64 tensors are implicitly converted to jax.float32 tensors?
    """
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
def jax_to_torch_dict(value: dict[str, jax.Array | Any], /) -> dict[str, torch.Tensor | Any]:
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


class JaxToTorchMixin:
    """A mixin that just implements the step and reset that convert jax arrays into torch tensors.

    TODO: Eventually make this support dict / tuples observations and actions:
    - use the generic `jax_to_torch` function.
    - mark this class generic w.r.t. the type of observations and actions
    """

    env: gymnasium.Env[jax.Array, jax.Array] | VectorEnv[jax.Array, jax.Array]

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, dict[Any, Any]
    ]:
        jax_action = torch_to_jax_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)
        torch_obs = jax_to_torch_tensor(obs)
        assert isinstance(reward, jax.Array)
        torch_reward = jax_to_torch_tensor(reward)

        if isinstance(terminated, bool):
            torch_terminated = torch.tensor(
                terminated, dtype=torch.bool, device=torch_reward.device
            )
        else:
            assert isinstance(terminated, jax.Array)
            torch_terminated = jax_to_torch_tensor(terminated)

        if isinstance(truncated, bool):
            torch_truncated = torch.tensor(truncated, dtype=torch.bool, device=torch_reward.device)
        else:
            assert isinstance(truncated, jax.Array)
            torch_truncated = jax_to_torch_tensor(truncated)

        # Brax has terminated and truncated as 0. and 1., here we convert them to bools instead.
        if torch_terminated.dtype != torch.bool:
            torch_terminated = torch_terminated.bool()
        if torch_truncated.dtype != torch.bool:
            torch_truncated = torch_truncated.bool()

        torch_info = jax_to_torch(info)
        return torch_obs, torch_reward, torch_terminated, torch_truncated, torch_info  # type: ignore

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        obs, info = self.env.reset(seed=seed, options=options)
        torch_obs = jax_to_torch_tensor(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info


def get_torch_device_from_jax_array(array: jax.Array) -> torch.device:
    jax_device = array.devices()
    assert len(jax_device) == 1
    jax_device_str = str(jax_device.pop())
    return get_torch_device_from_jax_device(jax_device_str)


def get_torch_device_from_jax_device(jax_device: str | jax.Device) -> torch.device:
    jax_device = str(jax_device)
    if jax_device.startswith("cuda"):
        device_type, _, index = jax_device.partition(":")
        assert index.isdigit()
        return torch.device(device_type, int(index))
    return torch.device("cpu")


def get_backend_from_torch_device(device: torch.device) -> str:
    if device.type == "cuda":
        return "gpu"
    if jax.default_backend() == "tpu":
        return "tpu"
    return "cpu"


def jit[C: Callable, **P](
    c: C, _fn: Callable[Concatenate[C, P], Any] = jax.jit, *args: P.args, **kwargs: P.kwargs
) -> C:
    # Fix `jax.jit` so it preserves the jit-ed function's signature and docstring.
    return _fn(c, *args, **kwargs)


def chexify[C: Callable, **P](
    c: C, _fn: Callable[Concatenate[C, P], Any] = chex.chexify, *args: P.args, **kwargs: P.kwargs
) -> C:
    # Fix `chex.chexify` so it preserves the jit-ed function's signature and docstring.
    return _fn(c, *args, **kwargs)

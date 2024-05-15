from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, overload

import jax
import numpy as np
import torch
from torch import Tensor

from project.utils.types import is_sequence_of

from .types import ActorOutput, Episode


@overload
def stack(values: list[Tensor]) -> Tensor: ...


@overload
def stack(values: list[jax.Array]) -> jax.Array: ...


@overload
def stack(values: list[np.ndarray]) -> np.ndarray: ...


@overload
def stack(values: list[dict]) -> dict: ...


@overload
def stack(values: list[int] | list[float] | list[bool]) -> np.ndarray: ...


def stack(
    values: list[Tensor]
    | list[jax.Array]
    | list[np.ndarray]
    | list[dict]
    | list[int]
    | list[float]
    | list[bool],
) -> Tensor | jax.Array | np.ndarray | dict:
    """Generic function for stacking lists of values into tensors or arrays.

    Also supports stacking dictionaries. This avoids creating new tensors, and keeps the return
    types as close to the input types as possible, to avoid unintentionally moving data between
    devices.
    """
    if is_sequence_of(values, dict):
        # NOTE: weird bug in the type checker? Doesn't understand that `values` is a list[Tensor].
        return stack_dicts(values)

    if is_sequence_of(values, jax.Array):
        return jax.numpy.stack(values)

    if is_sequence_of(values, np.ndarray):
        return np.stack(values)

    # Stack a list of tensors with different shapes in a single dimension:
    if is_sequence_of(values, Tensor):
        if not isinstance(values, list):
            values = list(values)
        if any(v.is_nested or v.shape != values[0].shape for v in values):
            return torch.nested.as_nested_tensor(values)
        return torch.stack(values)

    if is_sequence_of(values, (int, float, bool)):
        # idea: return a np.ndarray to emphasize that we won't move stuff between devices unless
        # necessary. For now though we return a CPU tensor.
        return np.array(values)
    raise NotImplementedError(f"Don't know how to stack these values: {values}")


def stack_dicts[M: Mapping](values: Sequence[M]) -> M:
    if not isinstance(values, list):
        values = list(values)
    all_keys = set().union(*[v.keys() for v in values])
    return_type = type(values[0])

    items = {}
    for key in all_keys:
        item_values = [v[key] for v in values]
        if isinstance(item_values[0], dict):
            assert is_sequence_of(item_values, dict)
            items[key] = stack_dicts(item_values)
        else:
            items[key] = stack(item_values)
    result = return_type(**items)
    return result


def stack_episode(
    observations: list[Tensor],
    actions: list[Tensor],
    rewards: list[Tensor],
    infos: list[dict],
    truncated: bool,
    terminated: bool,
    actor_outputs: list[ActorOutput],
    final_observation: Tensor | None = None,
    final_info: dict | None = None,
    environment_index: int | None = None,
) -> Episode[ActorOutput]:
    """Stacks the lists of items at each step into an Episode dict containing tensors."""

    return Episode(
        observations=stack(observations),
        actions=stack(actions),
        rewards=stack(rewards),  # RecordEpisodeStatistics wrapper needs np.ndarray rewards.
        infos=infos,  # todo: do we want to stack episode info dicts?
        truncated=truncated,
        terminated=terminated,
        actor_outputs=stack(actor_outputs),
        final_observation=final_observation,
        final_info=final_info,
        environment_index=environment_index,
    )


def _get_device(values: Any) -> torch.device:
    """Retrieve the Device of the first found Tensor in `values`."""

    def _get_device(value: Tensor | Any) -> torch.device | None:
        if isinstance(value, Tensor):
            return value.device
        if isinstance(value, dict):
            for k, v in value.items():
                device = _get_device(v)
                if device is not None:
                    return device
            return None
        if isinstance(value, list | tuple):
            for v in value:
                device = _get_device(v)
                if device is not None:
                    return device
            return None
        return None

    device = _get_device(values)
    if device is None:
        raise ValueError("There are no tensors in values, can't determine the device!")
    return device

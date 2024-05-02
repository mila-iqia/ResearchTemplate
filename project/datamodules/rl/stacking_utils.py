from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, TypeGuard, Unpack, overload

import torch
from torch import Tensor

from project.utils.types import is_sequence_of

from .rl_types import ActorOutput, Episode, UnstackedEpisode


@overload
def stack(values: list[Tensor], **kwargs) -> Tensor: ...


@overload
def stack(values: list[ActorOutput], **kwargs) -> ActorOutput: ...


def stack(values: list[Tensor] | list[ActorOutput], **kwargs) -> Tensor | ActorOutput:
    """Stack lists of tensors into a Tensor or dicts of tensors into a dict of stacked tensors."""
    if isinstance(values[0], dict):
        # NOTE: weird bug in the type checker? Doesn't understand that `values` is a list[Tensor].
        return stack_dicts(values, **kwargs)  # type: ignore
    assert isinstance(values[0], Tensor)
    if contains_only_tensors(values) and any(
        isinstance(v, Tensor) and (v.is_nested or v.shape != values[0].shape) for v in values
    ):
        return torch.nested.as_nested_tensor(values)
    return torch.stack(values, **kwargs)  # type: ignore


def contains_only_tensors(values: list[Any]) -> TypeGuard[list[Tensor]]:
    return all(isinstance(v, Tensor) for v in values)


def stack_dicts[**P, T: Tensor, V](
    values: Sequence[Mapping[str, T]],
    stack_fn: Callable[Concatenate[list[T], P], V] = stack,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Mapping[str, V]:
    values = list(values)
    all_keys = set().union(*[v.keys() for v in values])
    return_type: type[Mapping[str, V]] = type(values[0])  # type: ignore

    items = {}
    for key in all_keys:
        item_values = [v[key] for v in values]
        if isinstance(item_values[0], dict):
            assert is_sequence_of(item_values, dict)
            items[key] = stack_dicts(item_values, stack_fn, *args, **kwargs)
        else:
            items[key] = stack_fn(item_values, *args, **kwargs)
    result = return_type(**items)
    return result


def stack_episode(
    **episode: Unpack[UnstackedEpisode[ActorOutput]],
) -> Episode[ActorOutput]:
    """Stacks the lists of items at each step into an Episode dict containing tensors."""
    device = _get_device(episode)
    return Episode(
        observations=stack(episode["observations"]),
        actions=stack(episode["actions"]),
        rewards=torch.as_tensor(episode["rewards"], device=device),
        infos=episode["infos"],
        truncated=episode["truncated"],
        terminated=episode["terminated"],
        actor_outputs=stack(episode["actor_outputs"]),
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

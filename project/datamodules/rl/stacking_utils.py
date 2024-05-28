import functools
from collections.abc import Callable, Mapping, Sequence
from logging import getLogger as get_logger
from typing import Any, overload

import jax
import numpy as np
import torch
from torch import Tensor

from project.datamodules.rl.wrappers.jax_torch_interop import jax_to_torch_tensor
from project.utils.types import is_sequence_of

from .types import ActorOutput, Episode

logger = get_logger(__name__)


_stacking_fns: dict[type, Callable] = {
    np.ndarray: np.stack,
    # For scalars we return a np.ndarray to emphasize that we won't move stuff between devices unless
    # necessary.
    int: np.array,
    float: np.array,
    bool: np.array,
    type(None): np.array,
}
"""Mapping from item type to the function to use to stack these items."""


def register_stacking_fn[T](item_type: type[T]):
    def _wrapper[C: Callable[[Sequence], Any]](fn: C) -> C:
        _stacking_fns[item_type] = fn
        return fn

    return _wrapper


@overload
def stack(values: list[Tensor]) -> Tensor: ...


@overload
def stack(values: list[jax.Array]) -> jax.Array: ...


@overload
def stack(values: list[np.ndarray]) -> np.ndarray: ...


@overload
def stack[M: Mapping](values: list[M]) -> M: ...


@overload
def stack(values: list[int] | list[float] | list[bool]) -> np.ndarray: ...


def stack(
    values: list[Tensor]
    | list[jax.Array]
    | list[np.ndarray]
    | list[Mapping]
    | list[int]
    | list[float]
    | list[bool],
) -> Tensor | jax.Array | np.ndarray | Mapping:
    """Generic function for stacking lists of values into tensors or arrays.

    Also supports stacking dictionaries. This avoids creating new tensors, and keeps the return
    types as close to the input types as possible, to avoid unintentionally moving data between
    devices.
    """
    # Sort the functions so that the narrowest type that fits is used first.
    for item_type, function in sorted(
        _stacking_fns.items(), key=lambda x: len(x[0].mro()), reverse=True
    ):
        if is_sequence_of(values, item_type):
            return function(values)

    raise NotImplementedError(f"Don't know how to stack these values: {values}")


@register_stacking_fn(torch.Tensor)
def stack_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack a tensors with a possibly different length in a single dimension."""
    if not isinstance(values, list):
        values = list(values)
    if any(v.is_nested or v.shape != values[0].shape for v in values):
        return torch.nested.as_nested_tensor(values)
    return torch.stack(values)


@register_stacking_fn(jax.Array)
def stack_jax_arrays(values: Sequence[jax.Array]) -> torch.Tensor | jax.Array:
    """Stack jax arrays into a single jax array if they have the same shape, otherwise returns a
    Torch nested tensor."""
    if all(value.shape == values[0].shape for value in values):
        return jax.numpy.stack(values)
    return torch.nested.as_nested_tensor([jax_to_torch_tensor(value) for value in values])


@register_stacking_fn(Mapping)
def stack_mappings[M: Mapping](values: Sequence[M]) -> M:
    if not isinstance(values, list):
        values = list(values)
    all_keys = set().union(*[v.keys() for v in values])
    return_type = type(values[0])

    items = {}
    for key in all_keys:
        item_values = [v[key] for v in values]
        if isinstance(item_values[0], dict):
            assert is_sequence_of(item_values, dict)
            items[key] = stack_mappings(item_values)
        else:
            items[key] = stack(item_values)
    result = return_type(**items)
    return result


@register_stacking_fn(torch.distributions.Distribution)
def stack_distributions[D](values: Sequence[D]) -> D:
    """Stack multiple distributions."""
    raise NotImplementedError(f"Don't know how to stack distributions of type {type(values[0])}")


@register_stacking_fn(torch.distributions.Independent)
def stack_independent_distributions(
    values: Sequence[torch.distributions.Independent],
) -> torch.distributions.Independent:
    n_batch_dims = values[0].reinterpreted_batch_ndims
    assert all(d.reinterpreted_batch_ndims == n_batch_dims for d in values)
    return torch.distributions.Independent(
        stack([d.base_dist for d in values]), reinterpreted_batch_ndims=n_batch_dims + 1
    )


@register_stacking_fn(torch.distributions.Normal)
def stack_normal_distributions[D: torch.distributions.Normal](
    values: Sequence[D],
) -> torch.distributions.Independent:
    assert len(set([type(d) for d in values])) == 1
    loc = stack([v.loc for v in values])
    scale = stack([v.scale for v in values])
    return torch.distributions.Independent(
        type(values[0])(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1,
    )


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


@functools.singledispatch
def unstack(v: Any, /, *, n_slices: int) -> list[Any]:
    raise NotImplementedError(
        f"Don't know how to slice values of type {type(v)} into {n_slices} slices."
    )


@unstack.register
def _unstack_dict(v: Mapping, n_slices: int) -> list[Mapping]:
    keys = list(v.keys())
    unstacked_values: list[list] = [unstack(v, n_slices=n_slices) for v in v.values()]
    return [{k: v[i] for k, v in zip(keys, unstacked_values)} for i in range(n_slices)]


@unstack.register(type(None))
@unstack.register(int | float | str | bool)
def _unstack_shallow_copy[T](v: T, n_slices: int) -> list[T]:
    return [v for _ in range(n_slices)]


@unstack.register(Tensor | np.ndarray | jax.Array)
def _unstack_arraylike[T: Tensor | np.ndarray | jax.Array](v: T, n_slices: int) -> list[T]:
    assert v.shape[0] == n_slices
    return list(v)
    # duplicate the value.
    return [v for _ in range(n_slices)]


@unstack.register(list)
def _unstack_list[V](v: list[V], n_slices: int) -> list[V]:
    assert len(v) == n_slices
    return v.copy()


@unstack.register(torch.distributions.Categorical)
def _unstack_categorical[D: torch.distributions.Categorical](v: D, n_slices: int) -> list[D]:
    return [type(v)(logits=v.logits[i]) for i in range(n_slices)]


@unstack.register(torch.distributions.Normal)
def _unstack_normal[D: torch.distributions.Normal](v: D, n_slices: int) -> list[D]:
    loc = unstack(v.loc, n_slices=n_slices)
    scale = unstack(v.scale, n_slices=n_slices)
    return [type(v)(loc=loc[i], scale=scale[i]) for i in range(n_slices)]


@unstack.register(torch.distributions.Independent)
def _unstack_independent_distributions(
    v: torch.distributions.Independent, n_slices: int
) -> list[torch.distributions.Independent] | list[torch.distributions.Distribution]:
    if v.reinterpreted_batch_ndims == 1:
        return unstack(v.base_dist, n_slices=n_slices)
    return [
        torch.distributions.Independent(
            d, reinterpreted_batch_ndims=v.reinterpreted_batch_ndims - 1
        )
        for d in unstack(v.base_dist, n_slices=n_slices)
    ]

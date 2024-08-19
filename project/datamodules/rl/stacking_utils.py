import functools
from collections.abc import Callable, Mapping, Sequence
from logging import getLogger as get_logger
from typing import Any, Generic, TypeVar, overload
from typing_extensions import ParamSpec

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

T = TypeVar("T")


def register_stacking_fn(item_type: type[T]):
    C = TypeVar("C", bound=Callable[[Sequence], Any])

    def _wrapper(fn: C) -> C:
        _stacking_fns[item_type] = fn
        return fn

    return _wrapper


@overload
def stack(values: list[Tensor]) -> Tensor: ...


@overload
def stack(values: list[jax.Array]) -> jax.Array: ...


@overload
def stack(values: list[np.ndarray]) -> np.ndarray: ...


M = TypeVar("M", bound=Mapping)


@overload
def stack(values: list[M]) -> M: ...


@overload
def stack(values: list[int] | list[float] | list[bool]) -> np.ndarray: ...


def stack(
    values: (
        list[Tensor]
        | list[jax.Array]
        | list[np.ndarray]
        | list[Mapping]
        | list[int]
        | list[float]
        | list[bool]
    ),
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
    return torch.nested.as_nested_tensor(
        [jax_to_torch_tensor(value) for value in values]
    )


M = TypeVar("M", bound=Mapping)


@register_stacking_fn(Mapping)
def stack_mappings(values: Sequence[M]) -> M:
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


D = TypeVar("D")


@register_stacking_fn(torch.distributions.Distribution)
def stack_distributions(values: Sequence[D]) -> D:
    """Stack multiple distributions."""
    raise NotImplementedError(
        f"Don't know how to stack distributions of type {type(values[0])}"
    )


@register_stacking_fn(torch.distributions.Independent)
def stack_independent_distributions(
    values: Sequence[torch.distributions.Independent],
) -> torch.distributions.Independent:
    n_batch_dims = values[0].reinterpreted_batch_ndims
    assert all(d.reinterpreted_batch_ndims == n_batch_dims for d in values)
    return torch.distributions.Independent(
        stack([d.base_dist for d in values]), reinterpreted_batch_ndims=n_batch_dims + 1
    )


N = TypeVar("N", bound=torch.distributions.Normal)


@register_stacking_fn(torch.distributions.Normal)
def stack_normal_distributions(
    values: Sequence[N],
) -> torch.distributions.Independent:
    assert len(set([type(d) for d in values])) == 1
    loc = stack([v.loc for v in values])
    scale = stack([v.scale for v in values])
    return torch.distributions.Independent(
        type(values[0])(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1,
    )


DistType = TypeVar("DistType", bound=torch.distributions.Distribution)
P = ParamSpec("P")


class NestedDistribution(torch.distributions.Distribution, Generic[DistType]):
    def __init__(
        self,
        dist_type: Callable[P, DistType],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        assert is_sequence_of(
            args, torch.Tensor
        ), "expected only nested tensors in args"
        _unbind_args = [arg.unbind() for arg in args]
        _values = kwargs.values()
        assert is_sequence_of(
            _values, torch.Tensor
        ), "expected only nested tensors in kwargs"
        unbind_kwargs = {k: v.unbind() for k, v in zip(kwargs.keys(), _values)}
        n_dists: int | None = None
        for arg in _unbind_args:
            if isinstance(arg, tuple):
                n_dists = len(arg)
                break
        else:
            for k, v in unbind_kwargs.items():
                if isinstance(v, tuple):
                    n_dists = len(v)
                    break

        if n_dists is None:
            raise ValueError(
                f"couldn't infer the number of distributions from {args=} and {kwargs=}"
            )

        args_for_each_dist = [
            tuple(arg_i[j] for arg_i in _unbind_args) for j in range(n_dists)
        ]
        kwargs_for_each_dist = [
            {k: v[j] for k, v in unbind_kwargs.items()} for j in range(n_dists)
        ]
        self._distributions: list[DistType] = [
            dist_type(*args, **kwargs)
            for arg, kwargs in zip(args_for_each_dist, kwargs_for_each_dist)
        ]
        batch_shape = torch.Size(
            [len(self._distributions), *self._distributions[0].batch_shape]
        )
        super().__init__(batch_shape=batch_shape, validate_args=False)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return torch.nested.as_nested_tensor(
            [dist.sample(sample_shape) for dist in self._distributions]
        )

    def log_prob(self, value: Tensor) -> Tensor:
        assert value.is_nested
        values = value.unbind()
        assert len(values) == len(self._distributions)
        return torch.nested.as_nested_tensor(
            [dist.log_prob(val) for dist, val in zip(self._distributions, values)]
        )


class NestedCategorical(NestedDistribution[torch.distributions.Categorical]):
    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            kwargs = {"probs": probs}
        else:
            kwargs = {"logits": logits}
        super().__init__(torch.distributions.Categorical, **kwargs)


@register_stacking_fn(torch.distributions.Categorical)
def stack_categorical_distributions(
    values: Sequence[torch.distributions.Categorical],
) -> torch.distributions.Categorical | NestedCategorical:
    assert len(set([type(d) for d in values])) == 1

    logits = stack([v.logits for v in values])
    if logits.is_nested:
        return NestedCategorical(logits=logits)
    else:
        return torch.distributions.Categorical(logits=logits)


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
    discount_factor: float | None = None,
) -> Episode[ActorOutput]:
    """Stacks the lists of items at each step into an Episode dict containing tensors."""
    stacked_rewards = stack(rewards)
    returns: Tensor | None = None
    if discount_factor is not None:
        from project.algorithms.rl_example.reinforce import get_returns

        returns = get_returns(stacked_rewards, gamma=discount_factor)

    return Episode(
        observations=stack(observations),
        actions=stack(actions),
        rewards=stacked_rewards,
        infos=infos,  # We don't stack the info dicts because the keys aren't necessarily consistent.
        truncated=truncated,
        terminated=terminated,
        actor_outputs=stack(actor_outputs),
        final_observation=final_observation,
        final_info=final_info,
        environment_index=environment_index,
        returns=returns,
        discount_factor=discount_factor,
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
@unstack.register(int)
@unstack.register(float)
@unstack.register(str)
@unstack.register(bool)
def _unstack_shallow_copy(v: T, n_slices: int) -> list[T]:
    return [v for _ in range(n_slices)]


Ten = TypeVar("Ten", bound=Tensor | np.ndarray | jax.Array)


@unstack.register(Tensor)
@unstack.register(np.ndarray)
@unstack.register(jax.Array)
def _unstack_arraylike(v: Ten, n_slices: int) -> list[Ten]:
    assert v.shape[0] == n_slices
    return list(v)
    # duplicate the value.
    return [v for _ in range(n_slices)]


V = TypeVar("V")


@unstack.register(list)
def _unstack_list(v: list[V], n_slices: int) -> list[V]:
    assert len(v) == n_slices
    return v.copy()


Cat = TypeVar("Cat", bound=torch.distributions.Categorical)


@unstack.register(torch.distributions.Categorical)
def _unstack_categorical(v: Cat, n_slices: int) -> list[Cat]:
    return [type(v)(logits=v.logits[i]) for i in range(n_slices)]


N = TypeVar("N", bound=torch.distributions.Normal)


@unstack.register(torch.distributions.Normal)
def _unstack_normal(v: N, n_slices: int) -> list[N]:
    loc = unstack(v.loc, n_slices=n_slices)
    scale = unstack(v.scale, n_slices=n_slices)
    return [type(v)(loc=loc[i], scale=scale[i]) for i in range(n_slices)]


D = TypeVar("D", bound=torch.distributions.Distribution)


@unstack.register(NestedDistribution)
def _unstack_nested_distribution(v: NestedDistribution[D], n_slices: int) -> list[D]:
    return v._distributions.copy()


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

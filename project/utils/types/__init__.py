from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, NewType, TypeGuard, Unpack

import annotated_types
from torch import Tensor
from typing_extensions import TypeVar, TypeVarTuple

from .protocols import Dataclass, DataModule, HasInputOutputShapes, Module

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
S = NewType("S", int)


# Types used with pydantic:
FloatBetween0And1 = Annotated[float, annotated_types.Ge(0), annotated_types.Le(1)]

OutT = TypeVar("OutT", default=Tensor, covariant=True)
Ts = TypeVarTuple("Ts", default=Unpack[tuple[Tensor, ...]])
T = TypeVar("T", default=Tensor)

type NestedDict[K, V] = dict[K, V | NestedDict[K, V]]
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]
type PyTree[T] = T | Iterable[PyTree[T]] | Mapping[Any, PyTree[T]]


def is_list_of[V](object: Any, item_type: type[V] | tuple[type[V], ...]) -> TypeGuard[list[V]]:
    """Used to check (and tell the type checker) that `object` is a list of items of this type."""
    return isinstance(object, list) and is_sequence_of(object, item_type)


def is_sequence_of[V](
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to check (and tell the type checker) that `object` is a sequence of items of this
    type."""
    return isinstance(object, Sequence) and all(isinstance(value, item_type) for value in object)


def is_mapping_of[K, V](
    object: Any, key_type: type[K], value_type: type[V]
) -> TypeGuard[Mapping[K, V]]:
    """Used to check (and tell the type checker) that `object` is a mapping with keys and values of
    the given types."""
    return isinstance(object, Mapping) and all(
        isinstance(key, key_type) and isinstance(value, value_type)
        for key, value in object.items()
    )


__all__ = [
    "HasInputOutputShapes",
    "Module",
    "Dataclass",
    "DataModule",
]

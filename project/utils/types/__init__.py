from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import (
    Any,
    Literal,
    NewType,
    TypeGuard,
    Unpack,
)

from torch import Tensor
from typing_extensions import (
    ParamSpec,
    TypeVar,
    TypeVarTuple,
)

from .protocols import Dataclass, DataModule, HasInputOutputShapes, Module

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
S = NewType("S", int)


# todo: Fix this. Why do we have these enums? Are they necessary? Could we use the same ones as PL if we wanted to?
# from lightning.pytorch.trainer.states import RunningStage as PhaseStr
# from lightning.pytorch.trainer.states import TrainerFn as StageStr

PhaseStr = Literal["train", "val", "test"]
"""The trainer phases.

TODO: There has to exist an enum for it somewhere in PyTorch Lightning.
"""

P = ParamSpec("P", default=[Tensor])
R = ParamSpec("R")
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

from __future__ import annotations

from collections.abc import Sequence
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

from .outputs import ClassificationOutputs, StepOutputDict
from .protocols import Dataclass, HasInputOutputShapes, Module

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
S = NewType("S", int)

StageStr = Literal["fit", "validate", "test", "predict"]
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


def is_sequence_of[V](
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to tell the type checker that all items in the sequence are of the given type."""
    try:
        return all(isinstance(value, item_type) for value in object)
    except TypeError:
        return False


__all__ = [
    "HasInputOutputShapes",
    "Module",
    "Dataclass",
    "StepOutputDict",
    "ClassificationOutputs",
]

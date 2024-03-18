"""Typing utilities, used to annotate the source code to help prevent bugs."""

from __future__ import annotations

import collections.abc
import dataclasses
import typing
from typing import (
    Any,
    ClassVar,
    Literal,
    Mapping,
    NewType,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from lightning import LightningDataModule
from torch import Tensor, nn
from typing_extensions import (
    ParamSpec,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Unpack,
)

from project.utils.types.HasInputOutputShapes import HasInputOutputShapes

if typing.TYPE_CHECKING:
    from project.datamodules.datamodule import DataModule

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
S = NewType("S", int)
DM = TypeVar("DM", bound=Union[LightningDataModule, "DataModule"])
ModuleType = TypeVar("ModuleType", bound=nn.Module)
ModuleType_co = TypeVar("ModuleType_co", bound=nn.Module, covariant=True)
StageStr = Literal["fit", "validate", "test", "predict"]
PhaseStr = Literal["train", "val", "test"]
"""The trainer phases.

TODO: There has to exist an enum for it somewhere in PyTorch Lightning.
"""

P = ParamSpec("P", default=[Tensor])
R = ParamSpec("R")
OutT = TypeVar("OutT", default=Tensor, covariant=True)
Ts = TypeVarTuple("Ts", default=Unpack[Tuple[Tensor, ...]])
T = TypeVar("T", default=Tensor)
K = TypeVar("K")
V = TypeVar("V")
NestedDict = Mapping[K, Union[V, "NestedDict[K, V]"]]


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


_dict_values = type({}.values())
_dict_keys = type({}.values())


def is_sequence_of(
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to tell the type checker that all items in the sequence are of the given type."""

    return (
        object,
        (list, tuple, collections.abc.Sequence, _dict_keys, _dict_values),
    ) and all(isinstance(v, item_type) for v in object)

    # output_shapes: tuple[tuple[int, ...] | None, ...] = ()


def has_input_output_shapes_set(network: nn.Module) -> TypeGuard[HasInputOutputShapes]:
    return bool(getattr(network, "input_shape", ())) and bool(
        getattr(network, "output_shape", ())
    )

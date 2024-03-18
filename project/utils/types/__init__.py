from __future__ import annotations

import typing
from typing import (
    Literal,
    Mapping,
    NewType,
    Tuple,
    Union,
)

from lightning import LightningDataModule
from torch import Tensor
from typing_extensions import (
    ParamSpec,
    TypeVar,
    TypeVarTuple,
    Unpack,
)

if typing.TYPE_CHECKING:
    from project.datamodules.datamodule import DataModule
from .outputs import ClassificationOutputs, StepOutputDict
from .protocols import HasInputOutputShapes, Module

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
S = NewType("S", int)
DM = TypeVar("DM", bound=Union[LightningDataModule, "DataModule"])

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

type NestedDict[K, V] = Mapping[K, Union[V, NestedDict[K, V]]]


__all__ = [
    "HasInputOutputShapes",
    "Module",
    "StepOutputDict",
    "ClassificationOutputs",
]

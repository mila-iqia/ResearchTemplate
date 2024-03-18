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
from torch import Tensor, nn
from typing_extensions import (
    ParamSpec,
    TypeVar,
    TypeVarTuple,
    Unpack,
)

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

from .HasInputOutputShapes import HasInputOutputShapes
from .Module import Module
from .outputs import ClassificationOutputs, StepOutputDict

__all__ = [
    "HasInputOutputShapes",
    "Module",
    "StepOutputDict",
    "ClassificationOutputs",
]

"""Typing utilities, used to annotate the source code to help prevent bugs."""
from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal, NewType, Protocol, TypedDict, TypeVar
from typing_extensions import Required
from lightning import LightningDataModule
from torch import Tensor, nn

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
DM = TypeVar("DM", bound=LightningDataModule)
ModuleType = TypeVar("ModuleType", bound=nn.Module)
ModuleType_co = TypeVar("ModuleType_co", bound=nn.Module, covariant=True)
StageStr = Literal["fit", "validate", "test", "predict"]
PhaseStr = Literal["train", "val", "test"]
"""The trainer phases.

TODO: There has to exist an enum for it somewhere in PyTorch Lightning.
"""


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


class StepOutputDict(TypedDict, total=False):
    """A dictionary that shows what an Algorithm should output from `training/val/test_step`."""

    loss: Tensor | float
    """Optional loss tensor that can be returned by those methods."""

    log: dict[str, Tensor | Any]
    """Optional dictionary of things to log at each step."""


class ClassificationOutputs(StepOutputDict):
    """The dictionary format that is minimally required to be returned from
    `training/val/test_step`."""

    logits: Required[Tensor]
    """The un-normalized logits."""

    y: Required[Tensor]
    """The class labels."""

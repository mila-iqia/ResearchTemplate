from typing import Protocol, TypedDict

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor
from typing_extensions import NotRequired, TypeVar

from project.utils.typing_utils import PyTree
from project.utils.typing_utils.protocols import DataModule, Module


class StepOutputDict(TypedDict, total=False):
    """A dictionary that shows what an Algorithm can output from
    `training/validation/test_step`."""

    loss: NotRequired[Tensor | float]
    """Optional loss tensor that can be returned by those methods."""


BatchType = TypeVar("BatchType", bound=PyTree[torch.Tensor], contravariant=True)
StepOutputType = TypeVar("StepOutputType", bound=StepOutputDict, covariant=True)


class Algorithm(Module, Protocol[BatchType, StepOutputType]):
    """Protocol that adds more type information to the `lightning.LightningModule` class.

    This adds some type information on top of the LightningModule class, namely:
    - `BatchType`: The type of batch that is produced by the dataloaders of the datamodule
    - `StepOutputType`, the output type created by the step methods.

    The networks themselves are created separately and passed as a constructor argument. This is
    meant to make it easier to compare different learning algorithms on the same network
    architecture.
    """

    datamodule: LightningDataModule | DataModule[BatchType]
    network: Module

    def __init__(
        self,
        *,
        datamodule: DataModule[BatchType],
        network: Module,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        self.trainer: Trainer

    training_step = LightningModule.training_step
    # validation_step = LightningModule.validation_step

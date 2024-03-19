from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from project.utils.types import PhaseStr, StepOutputDict
from project.utils.types.protocols import DataModule, Module
from project.utils.utils import get_device


class Algorithm[NetworkType: Module, BatchType](LightningModule, ABC):
    """Base class for a learning algorithm.

    This is an extension of the LightningModule class from PyTorch Lightning, with some common
    boilerplate code to keep the algorithm implementations as simple as possible.

    The networks themselves are created separately and passed as a constructor argument. This is
    meant to make it easier to compare different learning algorithms on the same network
    architecture.
    """

    @dataclass
    class HParams:
        """Hyper-parameters of the algorithm."""

    def __init__(
        self,
        datamodule: DataModule[BatchType],
        network: NetworkType,
        hp: Algorithm.HParams | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self._device = get_device(network)  # fix for `self.device` property which defaults to cpu.
        self.network = network
        self.hp = hp or self.HParams()
        self.trainer: Trainer

    def training_step(self, batch: BatchType, batch_index: int) -> StepOutputDict:
        """Performs a training step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="train")

    def validation_step(self, batch: BatchType, batch_index: int) -> StepOutputDict:
        """Performs a validation step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="val")

    def test_step(self, batch: BatchType, batch_index: int) -> StepOutputDict:
        """Performs a test step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="test")

    def shared_step(self, batch: BatchType, batch_index: int, phase: PhaseStr) -> StepOutputDict:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training across GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        # """Creates the optimizers and the learning rate schedulers."""'
        # super().configure_optimizers()
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass.

        Feel free to overwrite this to do whatever you'd like.
        """
        return self.network(x)

    def configure_callbacks(self) -> list[Callback]:
        """Use this to add some callbacks that should always be included with the model."""
        if getattr(self.hp, "use_scheduler", False) and self.trainer and self.trainer.logger:
            from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

            return [LearningRateMonitor()]
        return []

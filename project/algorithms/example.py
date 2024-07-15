"""Example of an algorithm, which is a Pytorch Lightning image classifier.

Uses regular backpropagation.
"""

import dataclasses
import functools
from logging import getLogger
from typing import Annotated, Any, Literal

import pydantic
import torch
from hydra_zen import instantiate
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback, EarlyStopping
from pydantic import NonNegativeInt, PositiveInt
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from project.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from project.configs.algorithm.lr_scheduler import CosineAnnealingLRConfig
from project.configs.algorithm.optimizer import AdamConfig
from project.datamodules.image_classification import ImageClassificationDataModule

logger = getLogger(__name__)

LRSchedulerConfig = Annotated[Any, pydantic.Field(default_factory=CosineAnnealingLRConfig)]


class ExampleAlgorithm(LightningModule):
    """Example learning algorithm for image classification."""

    @pydantic.dataclasses.dataclass(frozen=True)
    class HParams:
        """Hyper-Parameters."""

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: LRSchedulerConfig = dataclasses.field(
            default=CosineAnnealingLRConfig(T_max=85, eta_min=1e-5),
            metadata={"omegaconf_ignore": True},
        )

        lr_scheduler_interval: Literal["step", "epoch"] = "epoch"

        # Frequency of the LR scheduler. Set to 0 to disable the lr scheduler.
        lr_scheduler_frequency: NonNegativeInt = 1

        # Hyper-parameters for the optimizer
        optimizer: Any = AdamConfig(lr=3e-4)

        batch_size: PositiveInt = 128

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: NonNegativeInt = 0

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: torch.nn.Module,
        hp: HParams = HParams(),
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        self.hp = hp or self.HParams()

        # Used by Pytorch-Lightning to compute the input/output shapes of the network.
        self.example_input_array = torch.zeros(
            (datamodule.batch_size, *datamodule.dims), device=self.device
        )
        # Do a forward pass to initialize any lazy weights. This is necessary for distributed
        # training and to infer shapes.
        _ = self.network(self.example_input_array)

        # Save hyper-parameters.
        self.save_hyperparameters({"hp": dataclasses.asdict(self.hp)})

    def forward(self, input: Tensor) -> Tensor:
        logits = self.network(input)
        return logits

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="test")

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        self.log(f"{phase}/loss", loss.detach().mean())
        acc = logits.detach().argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/accuracy", acc)
        return {"loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self) -> dict:
        """Creates the optimizers and the LR scheduler (if needed)."""
        optimizer_partial: functools.partial[Optimizer]
        if isinstance(self.hp.optimizer, functools.partial):
            optimizer_partial = self.hp.optimizer
        else:
            optimizer_partial = instantiate(self.hp.optimizer)
        optimizer = optimizer_partial(self.parameters())
        optimizers: dict[str, Any] = {"optimizer": optimizer}

        lr_scheduler_partial: functools.partial[_LRScheduler]
        if isinstance(self.hp.lr_scheduler, functools.partial):
            lr_scheduler_partial = self.hp.lr_scheduler
        else:
            lr_scheduler_partial = instantiate(self.hp.lr_scheduler)

        if self.hp.lr_scheduler_frequency != 0:
            lr_scheduler = lr_scheduler_partial(optimizer)
            optimizers["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                # NOTE: These two keys are ignored if doing manual optimization.
                # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html#learning-rate-scheduling
                "interval": self.hp.lr_scheduler_interval,
                "frequency": self.hp.lr_scheduler_frequency,
            }
        return optimizers

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        callbacks.append(
            # Log some classification metrics. (This callback adds some metrics on this module).
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        )
        if self.hp.lr_scheduler_frequency != 0:
            from lightning.pytorch.callbacks import LearningRateMonitor

            callbacks.append(LearningRateMonitor())
        if self.hp.early_stopping_patience != 0:
            # If early stopping is enabled, add a Callback for it:
            callbacks.append(
                EarlyStopping(
                    "val/accuracy",
                    mode="max",
                    patience=self.hp.early_stopping_patience,
                    verbose=True,
                )
            )
        return callbacks

    @property
    def device(self) -> torch.device:
        """Small fixup for the `device` property in LightningModule, which is CPU by default."""
        if self._device.type == "cpu":
            self._device = next((p.device for p in self.parameters()), torch.device("cpu"))
        device = self._device
        # make this more explicit to always include the index
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", index=torch.cuda.current_device())
        return device

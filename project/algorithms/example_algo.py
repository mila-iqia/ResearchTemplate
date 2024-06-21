"""Example of an algorithm, which is a Pytorch Lightning image classifier.

Uses regular backpropagation.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from project.algorithms.bases.image_classification import (
    ClassificationOutputs,
    ImageClassificationAlgorithm,
)
from project.configs.algorithm.lr_scheduler import CosineAnnealingLRConfig
from project.configs.algorithm.optimizer import AdamConfig
from project.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.hydra_utils import instantiate
from project.utils.types import PhaseStr
from project.utils.types.protocols import Module

logger = getLogger(__name__)


class ExampleAlgorithm(ImageClassificationAlgorithm):
    """Example algorithm for image classification."""

    # TODO: Make this less specific to Image classification once we add other supervised learning
    # settings.

    @dataclass
    class HParams(ImageClassificationAlgorithm.HParams):
        """Hyper-Parameters of the baseline model."""

        # Arguments to be passed to the LR scheduler.
        lr_scheduler: CosineAnnealingLRConfig = CosineAnnealingLRConfig(T_max=85, eta_min=1e-5)

        lr_scheduler_interval: str = "epoch"

        # Frequency of the LR scheduler. Set to 0 to disable the lr scheduler.
        lr_scheduler_frequency: int = 1

        # Max number of training epochs in total.
        max_epochs: int = 90

        # Hyper-parameters for the forward optimizer
        # BUG: seems to be reproducible given a seed when using SGD, but not when using Adam!
        optimizer: AdamConfig = AdamConfig(lr=3e-4)

        # batch size
        batch_size: int = 128

        # Max number of epochs to train for without an improvement to the validation
        # accuracy before the training is stopped.
        early_stopping_patience: int = 0

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: Module[[Tensor], Tensor],
        hp: ExampleAlgorithm.HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.datamodule: ImageClassificationDataModule
        self.hp: ExampleAlgorithm.HParams
        self.automatic_optimization = True

        # Initialize any lazy weights.
        _ = self.network(self.example_input_array)
        self.save_hyperparameters({"hp": dataclasses.asdict(self.hp)})

    def forward(self, input: Tensor) -> Tensor:
        logits = self.network(input)
        return logits

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: PhaseStr,
    ) -> ClassificationOutputs:
        x, y = batch
        logits = self(x)
        # reduction=none to get the proper gradients in a backward pass when using multiple gpus.
        loss = F.cross_entropy(logits, y, reduction="none")
        self.log(f"{phase}/loss", loss.detach().mean())

        # probs = torch.softmax(logits, -1)
        # accuracy = getattr(self, f"{phase}_accuracy")
        # top5_accuracy = getattr(self, f"{phase}_top5_accuracy")

        # # TODO: It's a bit confusing, not sure if this is the right way to use this:
        # accuracy(probs, y)
        # top5_accuracy(probs, y)
        # prog_bar = phase == "train"
        # self.log(f"{phase}/accuracy", accuracy, prog_bar=prog_bar, sync_dist=True)
        # self.log(f"{phase}/top5_accuracy", top5_accuracy, prog_bar=prog_bar, sync_dist=True)
        return {"loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self) -> dict:
        """Creates the optimizers and the LR schedulers (if needed)."""
        optimizer_partial: functools.partial[Optimizer] = instantiate(self.hp.optimizer)
        lr_scheduler_partial: functools.partial[_LRScheduler] = instantiate(self.hp.lr_scheduler)
        optimizer = optimizer_partial(self.parameters())

        optimizers: dict[str, Any] = {"optimizer": optimizer}

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
        callbacks: list[Callback] = super().configure_callbacks()
        if self.hp.early_stopping_patience != 0:
            # If early stopping is enabled, add a PL Callback for it:
            callbacks.append(
                EarlyStopping(
                    "val/accuracy",
                    mode="max",
                    patience=self.hp.early_stopping_patience,
                    verbose=True,
                )
            )
        return callbacks

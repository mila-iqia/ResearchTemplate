"""Pytorch Lightning image classifier.

Uses regular backprop.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
from hydra_zen import instantiate
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import MulticlassAccuracy

from project.algorithms.bases.algorithm import Algorithm
from project.configs.algorithm.lr_scheduler import CosineAnnealingLRConfig
from project.configs.algorithm.optimizer import AdamConfig
from project.datamodules.image_classification import ImageClassificationDataModule
from project.utils.types import PhaseStr, StepOutputDict

logger = getLogger(__name__)


class Backprop(Algorithm):
    """Baseline model that uses normal backpropagation."""

    # TODO: Make this less specific to Image classification once we add other supervised learning
    # settings.

    @dataclass
    class HParams(Algorithm.HParams):
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
        network: nn.Module,
        hp: HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, hp=hp)
        self.datamodule: ImageClassificationDataModule
        self.hp = hp
        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        # TODO: Check if PL now moves the `example_input_array` to the right device automatically.
        # If possible, we'd like to remove any reference to the device from the algorithm.
        # device = get_device(self.network)
        self.example_input_array = torch.rand(
            [datamodule.batch_size, *datamodule.dims],
            # device=device,
        )
        num_classes: int = datamodule.num_classes

        # IDEA: Could use a dict of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: dist[str, Metrics]
        # NOTE: Need to have one per phase! Not 100% sure that I'm not forgetting a phase here.
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.train_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)

        self.hp: Backprop.HParams
        self.automatic_optimization = True

        # Initialize any lazy weights.
        _ = self.network(self.example_input_array)

        # TODO: Check that this works with the dataclasses.
        self.save_hyperparameters({"network_type": type(network), "hp": self.hp})

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        # Dummy forward pass, not used in practice. We just implement it so that PL can
        # display the input/output shapes of our networks.
        logits = self.network(input)
        return logits

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        x, y = batch
        logits = self(x)
        # NOTE: There's an issue with PyTorch-Lightning's precision plugin, it assumes that the
        # loss, if returned, is a scalar!

        # reduction=sum to get the proper gradients in a backward pass when using multiple gpus.
        loss = F.cross_entropy(logits, y, reduction="none")
        self.log(f"{phase}/local_loss", loss.detach().mean())

        total_loss = self.all_gather(loss, sync_grads=True)
        assert isinstance(total_loss, Tensor)
        loss = total_loss.mean()
        self.log(f"{phase}/loss", loss, rank_zero_only=True)

        probs = torch.softmax(logits, -1)
        accuracy = getattr(self, f"{phase}_accuracy")
        top5_accuracy = getattr(self, f"{phase}_top5_accuracy")

        # TODO: It's a bit confusing, not sure if this is the right way to use this:
        accuracy(probs, y)
        top5_accuracy(probs, y)
        prog_bar = phase == "train"
        self.log(f"{phase}/accuracy", accuracy, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{phase}/top5_accuracy", top5_accuracy, prog_bar=prog_bar, sync_dist=True)
        return {"loss": loss}

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

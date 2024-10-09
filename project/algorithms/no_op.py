from typing import Any, Literal

import torch
from lightning import Callback, LightningModule

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.utils.typing_utils.protocols import DataModule


class NoOp(LightningModule):
    """No-op algorithm that does no learning and is used to benchmark the dataloading speed."""

    def __init__(self, datamodule: DataModule):
        super().__init__()
        self.datamodule = datamodule
        # Set this so PyTorch-Lightning doesn't try to train the model using our 'loss'
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_index: int):
        return self.shared_step(batch, batch_index, "train")

    def validation_step(self, batch: Any, batch_index: int):
        return self.shared_step(batch, batch_index, "val")

    def test_step(self, batch: Any, batch_index: int):
        return self.shared_step(batch, batch_index, "test")

    def shared_step(
        self,
        batch: Any,
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ):
        fake_loss = torch.rand(1)
        self.log(f"{phase}/loss", fake_loss)
        return fake_loss

    def configure_callbacks(self) -> list[Callback]:
        return [MeasureSamplesPerSecondCallback()]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.123)

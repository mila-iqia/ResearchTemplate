from typing import Any

import torch
from lightning import Callback
from torch import nn

from project.algorithms.algorithm import Algorithm, StepOutputDict
from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.utils.types import PhaseStr
from project.utils.types.protocols import DataModule


class NoOp(Algorithm):
    """No-op algorithm that does no learning and is used to benchmark the dataloading speed."""

    def __init__(self, datamodule: DataModule, network: nn.Module):
        super().__init__(datamodule=datamodule, network=network)
        # Set this so PyTorch-Lightning doesn't try to train the model using our 'loss'
        self.automatic_optimization = False
        self.last_step_times: dict[PhaseStr, float] = {}

    def shared_step(
        self,
        batch: Any,
        batch_index: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        fake_loss = torch.rand(1)
        self.log(f"{phase}/loss", fake_loss)
        return {"loss": fake_loss}

    def configure_callbacks(self) -> list[Callback]:
        return super().configure_callbacks() + [MeasureSamplesPerSecondCallback()]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.123)

from typing import Any

from torch import nn

from project.algorithms.bases import Algorithm
from project.utils.types import PhaseStr, StepOutputDict
from project.utils.types.protocols import DataModule


class NoOp(Algorithm):
    """No-op algorithm that does no learning and is used to benchmark the dataloading speed."""

    def __init__(self, datamodule: DataModule, network: nn.Module):
        super().__init__(datamodule=datamodule, network=network)
        # Set this so PyTorch-Lightning doesn't try to train the model using our 'loss'
        self.automatic_optimization = False

    def shared_step(
        self,
        batch: Any,
        batch_index: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        return {"loss": 0.0}

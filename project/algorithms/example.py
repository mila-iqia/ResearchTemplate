"""Example of a simple algorithm for image classification.

This can be run from the command-line like so:

```console
python project/main.py algorithm=example
```
"""

import dataclasses
import functools
from logging import getLogger
from typing import Any, Literal

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from project.configs.algorithm.optimizer import AdamConfig
from project.datamodules.image_classification import ImageClassificationDataModule
from project.experiment import instantiate

logger = getLogger(__name__)


class ExampleAlgorithm(LightningModule):
    """Example learning algorithm for image classification."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: torch.nn.Module,
        optimizer_config: Any = AdamConfig(lr=3e-4),
    ):
        """Create a new instance of the algorithm.

        Parameters
        ----------
        datamodule: Object used to load train/val/test data. See the lightning docs for the \
            `LightningDataModule` class more info.
        network: The network to train.
        optimizer_config: Configuration options for the Optimizer.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        self.optimizer_config = optimizer_config
        assert dataclasses.is_dataclass(optimizer_config) or isinstance(
            optimizer_config, dict | DictConfig
        ), optimizer_config

        # Used by Pytorch-Lightning to compute the input/output shapes of the network.
        self.example_input_array = torch.zeros(
            (datamodule.batch_size, *datamodule.dims), device=self.device
        )
        # Do a forward pass to initialize any lazy weights. This is necessary for distributed
        # training and to infer shapes.
        _ = self.network(self.example_input_array)

        # Save hyper-parameters.
        self.save_hyperparameters(ignore=["datamodule", "network"])

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
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        self.log(f"{phase}/loss", loss.detach().mean())
        acc = logits.detach().argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/accuracy", acc)
        return {"loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self):
        optimizer_partial: functools.partial[Optimizer]
        # todo: why are there two cases here? CLI vs programmatically? Why are they different?
        if isinstance(self.optimizer_config, functools.partial):
            optimizer_partial = self.optimizer_config
        else:
            optimizer_partial = instantiate(self.optimizer_config)
        optimizer = optimizer_partial(self.parameters())
        return optimizer

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

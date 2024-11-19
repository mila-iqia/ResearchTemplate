"""Example of a simple algorithm for image classification.

This can be run from the command-line like so:

```console
python project/main.py algorithm=image_classification datamodule=cifar10
```
"""

import functools
from collections.abc import Sequence
from logging import getLogger
from typing import Literal

import hydra_zen
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from project.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from project.datamodules.image_classification import ImageClassificationDataModule
from project.utils.typing_utils import HydraConfigFor

logger = getLogger(__name__)


class ImageClassifier(LightningModule):
    """Example learning algorithm for image classification."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: HydraConfigFor[torch.nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        init_seed: int = 42,
    ):
        """Create a new instance of the algorithm.

        Parameters:
            datamodule: Object used to load train/val/test data.
                See the lightning docs for [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule]
                for more info.
            network:
                The config of the network to instantiate and train.
            optimizer: The config for the Optimizer. Instantiating this will return a function \
                (a [functools.partial][]) that will create the Optimizer given the hyper-parameters.
            init_seed: The seed to use when initializing the weights of the network.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Save hyper-parameters.
        self.save_hyperparameters(ignore=["datamodule"])
        # Used by Pytorch-Lightning to compute the input/output shapes of the network.

        self.network: torch.nn.Module | None = None

    def configure_model(self):
        # Save this for PyTorch-Lightning to infer the input/output shapes of the network.
        self.example_input_array = torch.zeros((self.datamodule.batch_size, *self.datamodule.dims))
        with torch.random.fork_rng():
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)
            if any(torch.nn.parameter.is_lazy(p) for p in self.network.parameters()):
                # Do a forward pass to initialize any lazy weights. This is necessary for
                # distributed training and to infer shapes.
                _ = self.network(self.example_input_array)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the network."""
        assert self.network is not None
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
        """Creates the optimizers.

        See [`lightning.pytorch.core.LightningModule.configure_optimizers`][] for more information.
        """
        # Instantiate the optimizer config into a functools.partial object.
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        # Call the functools.partial object, passing the parameters as an argument.
        optimizer = optimizer_partial(self.parameters())
        # This then returns the optimizer.
        return optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Creates callbacks to be used by default during training."""
        return [
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        ]

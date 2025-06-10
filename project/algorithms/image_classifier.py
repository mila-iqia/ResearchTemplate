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
from typing_extensions import override

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.typing_utils import HydraConfigFor

logger = getLogger(__name__)


class ImageClassifier(LightningModule):
    """Example learning algorithm for image classification."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: torch.nn.Module | HydraConfigFor[torch.nn.Module],
        optimizer: functools.partial[Optimizer] | HydraConfigFor[functools.partial[Optimizer]],
        init_seed: int = 42,
    ):
        """Create a new instance of the algorithm.

        Parameters:
            datamodule: Object used to load train/val/test data.
                See the lightning docs for [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule]
                for more info.
            network:
                The network to instantiate and train, or a Hydra config that returns a network \
                when instantiated.
            optimizer: A function that returns an optimizer given parameters, or a Hydra config \
                that creates such a function when instantiated.
            init_seed: The seed to set while instantiating the network from its config. This only \
                has an effect if the network is a Hydra config, and not an already instantiated.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Save hyper-parameters.
        self.save_hyperparameters(
            ignore=["datamodule"]
            # Ignore those if they are already instantiated objects, otherwise lightning will try
            # to serialize them to yaml, which will be very slow and may fail.
            + (["network"] if isinstance(network, torch.nn.Module) else [])
            + (["optimizer"] if isinstance(optimizer, functools.partial) else [])
        )
        # Used by Pytorch-Lightning to compute the input/output shapes of the network.

        self.network: torch.nn.Module | None = (
            network if isinstance(network, torch.nn.Module) else None
        )
        self.logits_pinned: torch.Tensor | None = None  # type: Tensor | None
        self.labels_pinned: torch.Tensor | None = None  # type: Tensor | None

    def configure_model(self):
        # Save this for PyTorch-Lightning to infer the input/output shapes of the network.
        if self.network is not None:
            logger.info("Network is already instantiated.")
            return
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

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="train")

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="val")

    @override
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
        loss_mean = loss.detach().mean()
        acc = logits.detach().argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/loss", loss_mean)
        self.log(f"{phase}/accuracy", acc)
        return {"loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self):
        """Creates the optimizers.

        See [`lightning.pytorch.core.LightningModule.configure_optimizers`][] for more information.
        """
        if isinstance(self.optimizer_config, functools.partial):
            optimizer_partial = self.optimizer_config
        else:
            # Instantiate the optimizer config into a functools.partial object.
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        # Call the functools.partial object, passing the parameters as an argument.
        optimizer = optimizer_partial(self.parameters())
        # This then returns the optimizer.
        return optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Creates callbacks to be used by default during training."""
        return [
            MeasureSamplesPerSecondCallback(),
            # Uncomment to log top_k accuracy metrics:
            # ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        ]

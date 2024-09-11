"""Example of a simple algorithm for image classification.

This can be run from the command-line like so:

```console
python project/main.py algorithm=example
```
"""

from logging import getLogger
from typing import Literal

import torch
from hydra_zen.typing import Builds, PartialBuilds
from lightning import LightningModule
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
        network: Builds[type[torch.nn.Module]],
        optimizer: PartialBuilds[type[Optimizer]] = AdamConfig(lr=3e-4),
        init_seed: int = 42,
    ):
        """Create a new instance of the algorithm.

        Parameters
        ----------
        datamodule: Object used to load train/val/test data. See the lightning docs for the \
            `LightningDataModule` class more info.
        network: The config of the network to instantiate and train.
        optimizer: Configuration options for the Optimizer. Note that this is an optimizer.
        init_seed: The seed to use when initializing the weights of the network.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Save hyper-parameters.
        self.save_hyperparameters(
            {
                "network_config": self.network_config,
                "optimizer_config": self.optimizer_config,
                "init_seed": init_seed,
            }
        )

        # Save hyper-parameters.
        self.save_hyperparameters(ignore=["datamodule", "network"])

        # Small fix for the `device` property in LightningModule, which is CPU by default.
        self._device = next((p.device for p in self.parameters()), torch.device("cpu"))
        # Used by Pytorch-Lightning to compute the input/output shapes of the network.
        self.example_input_array = torch.zeros(
            (datamodule.batch_size, *datamodule.dims), device=self.device
        )

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            self.network = instantiate(self.network_config)

            if any(torch.nn.parameter.is_lazy(p) for p in self.network.parameters()):
                # Do a forward pass to initialize any lazy weights. This is necessary for distributed
                # training and to infer shapes.
                _ = self.network(self.example_input_array)

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
        # Instantiate the optimizer config into a functools.partial object.
        optimizer_partial = instantiate(self.optimizer_config)
        # Call the functools.partial object, passing the parameters as an argument.
        optimizer = optimizer_partial(self.parameters())
        # This then returns the optimizer.
        return optimizer

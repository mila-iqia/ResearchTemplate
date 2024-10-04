import dataclasses
import logging
import os
from typing import Literal

import chex
import flax.linen
import jax
import rich
import rich.logging
import torch
import torch.distributed
from lightning import Callback, LightningModule, Trainer
from torch_jax_interop import WrappedJaxFunction, torch_to_jax

from project.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.typing_utils.protocols import ClassificationDataModule


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))


class CNN(flax.linen.Module):
    """A simple CNN model.

    Taken from https://flax.readthedocs.io/en/latest/quick_start.html#define-network
    """

    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x: jax.Array):
        x = to_channels_last(x)
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = flatten(x)
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class JaxFcNet(flax.linen.Module):
    num_classes: int = 10
    num_features: int = 256

    @flax.linen.compact
    def __call__(self, x: jax.Array, forward_rng: chex.PRNGKey | None = None):
        # x = flatten(x)
        x = flax.linen.Dense(features=self.num_features)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class JaxExample(LightningModule):
    """Example of a learning algorithm (`LightningModule`) that uses Jax.

    In this case, the network is a flax.linen.Module, and its forward and backward passes are
    written in Jax, and the loss function is in pytorch.
    """

    @dataclasses.dataclass(frozen=True)
    class HParams:
        """Hyper-parameters of the algo."""

        lr: float = 1e-3
        seed: int = 123
        debug: bool = True

    def __init__(
        self,
        *,
        network: flax.linen.Module,
        datamodule: ImageClassificationDataModule,
        hp: HParams = HParams(),
    ):
        super().__init__()
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        self.datamodule = datamodule
        self.hp = hp or self.HParams()

        example_input = torch.zeros(
            (datamodule.batch_size, *datamodule.dims),
            device=self.device,
        )
        # Initialize the jax parameters with a forward pass.
        params = network.init(jax.random.key(self.hp.seed), x=torch_to_jax(example_input))

        # Wrap the jax network into a nn.Module:
        self.network = WrappedJaxFunction(
            jax_function=jax.jit(network.apply) if not self.hp.debug else network.apply,
            jax_params=params,
            # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
            # Invalid device pointer when trying to share the CUDA tensors that come from jax.
            clone_params=True,
            has_aux=False,
        )

        self.example_input_array = example_input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.network(input)
        return logits

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        return self.shared_step(batch, batch_index=batch_index, phase="test")

    def shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ):
        x, y = batch
        assert not x.requires_grad
        logits = self.network(x)
        assert isinstance(logits, torch.Tensor)
        # In this example we use a jax "encoder" network and a PyTorch loss function, but we could
        # also just as easily have done the whole forward and backward pass in jax if we wanted to.
        loss = torch.nn.functional.cross_entropy(logits, target=y, reduction="mean")
        acc = logits.argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/acc", acc, prog_bar=True, sync_dist=True)
        return {"loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)

    def configure_callbacks(self) -> list[Callback]:
        assert isinstance(self.datamodule, ClassificationDataModule)
        return [
            MeasureSamplesPerSecondCallback(),
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes),
        ]


# Register a handler function to "convert" `torch.nn.Parameter`s to jax Arrays: they can be viewed
# as jax Arrays by just viewing their data as a jax array.
# TODO: move this to the torch_jax_interop package?
@torch_to_jax.register(torch.nn.Parameter)
def _parameter_to_jax_array(value: torch.nn.Parameter) -> jax.Array:
    return torch_to_jax(value.data)


def is_channels_first(shape: tuple[int, ...]) -> bool:
    if len(shape) == 4:
        return is_channels_first(shape[1:])
    if len(shape) != 3:
        return False
    return (shape[0] in (1, 3) and shape[1] not in {1, 3} and shape[2] not in {1, 3}) or (
        shape[0] < min(shape[1], shape[2])
    )


def is_channels_last(shape: tuple[int, ...]) -> bool:
    if len(shape) == 4:
        return is_channels_last(shape[1:])
    if len(shape) != 3:
        return False
    return (shape[2] in (1, 3) and shape[0] not in {1, 3} and shape[1] not in {1, 3}) or (
        shape[2] < min(shape[0], shape[1])
    )


def to_channels_last(x: jax.Array) -> jax.Array:
    shape = tuple(x.shape)
    if is_channels_last(shape):
        return x
    if not is_channels_first(shape):
        return x
    if x.ndim == 3:
        return x.transpose(1, 2, 0)
    assert x.ndim == 4
    return x.transpose(0, 2, 3, 1)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[rich.logging.RichHandler()]
    )
    from lightning.pytorch.callbacks import RichProgressBar

    trainer = Trainer(
        devices="auto",
        max_epochs=10,
        accelerator="auto",
        callbacks=[RichProgressBar()],
    )
    datamodule = MNISTDataModule(num_workers=4, batch_size=512)
    network = CNN(num_classes=datamodule.num_classes)

    model = JaxExample(network=network, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    ...


if __name__ == "__main__":
    main()
    print("Done!")

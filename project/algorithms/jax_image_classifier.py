import functools
import logging
from typing import Literal

import flax.linen
import hydra_zen
import jax
import rich
import rich.logging
import torch
import torch.distributed
from lightning import Callback, LightningModule, Trainer
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch_jax_interop import WrappedJaxFunction, torch_to_jax

from project.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.typing_utils import HydraConfigFor


def flatten(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0], -1))


class JaxCNN(flax.linen.Module):
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
    def __call__(self, x: jax.Array):
        x = flatten(x)
        x = flax.linen.Dense(features=self.num_features)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class JaxImageClassifier(LightningModule):
    """Example of a learning algorithm (`LightningModule`) that uses Jax.

    In this case, the network is a flax.linen.Module, and its forward and backward passes are
    written in Jax, and the loss function is in pytorch.
    """

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: HydraConfigFor[flax.linen.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        init_seed: int = 123,
        debug: bool = True,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed
        self.debug = debug

        # Create the jax network (safe to do even on CPU here).
        self.jax_network: flax.linen.Module = hydra_zen.instantiate(self.network_config)
        # We'll instantiate the parameters and the torch wrapper around the jax network in
        # `configure_model` so the weights are directly on the GPU.
        self.network: torch.nn.Module | None = None
        self.save_hyperparameters(ignore=["datamodule"])

    def configure_model(self):
        example_input = torch.zeros(
            (self.datamodule.batch_size, *self.datamodule.dims),
        )
        # Save this for PyTorch-Lightning to infer the input/output shapes of the network.
        self.example_input_array = example_input

        # Initialize the jax parameters with a forward pass.
        jax_params = self.jax_network.init(
            jax.random.key(self.init_seed), torch_to_jax(example_input)
        )

        jax_network_forward = self.jax_network.apply
        if not self.debug:
            jax_network_forward = jax.jit(jax_network_forward)

        # Wrap the jax network into a nn.Module:
        self.network = WrappedJaxFunction(
            jax_function=jax_network_forward,
            jax_params=jax_params,
            # Need to call .clone() when doing distributed training, otherwise we get a RuntimeError:
            # Invalid device pointer when trying to share the CUDA tensors that come from jax.
            clone_params=True,
            has_aux=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.network is not None
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
        # This is the same thing as the `ImageClassifier.shared_step`!
        x, y = batch
        assert not x.requires_grad
        # This calls self.forward, and is preferable to calling self.network directly, since it
        # allows forward hooks to be called. This is useful for example when testing or debugging.
        logits = self(x)

        assert isinstance(logits, torch.Tensor)
        # In this example we use a jax "encoder" network and a PyTorch loss function, but we could
        # also just as easily have done the whole forward and backward pass in jax if we wanted to.
        loss = F.cross_entropy(logits, y, reduction="mean")
        acc = logits.argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/acc", acc, prog_bar=True, sync_dist=True)
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

    def configure_callbacks(self) -> list[Callback]:
        assert isinstance(self.datamodule, ImageClassificationDataModule)
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


def demo(**trainer_kwargs):
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[rich.logging.RichHandler()]
    )
    from lightning.pytorch.callbacks import RichProgressBar

    trainer = Trainer(
        **trainer_kwargs,
        accelerator="auto",
        callbacks=[RichProgressBar()],
    )
    datamodule = MNISTDataModule(num_workers=4, batch_size=64)
    network = JaxCNN(num_classes=datamodule.num_classes)
    optimizer = functools.partial(torch.optim.SGD, lr=0.01)  # type: ignore
    model = JaxImageClassifier(
        datamodule=datamodule,
        network=hydra_zen.just(network),  # type: ignore
        optimizer=hydra_zen.just(optimizer),  # type: ignore
    )
    trainer.fit(model, datamodule=datamodule)

    ...


if __name__ == "__main__":
    demo()
    print("Done!")

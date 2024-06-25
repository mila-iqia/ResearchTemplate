from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from project.algorithms.algorithm import Algorithm
from project.algorithms.callbacks.classification_metrics import (
    ClassificationMetricsCallback,
    ClassificationOutputs,
)
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.types import PhaseStr


class ManualGradientsExample(Algorithm):
    """Example of an algorithm that calculates the gradients manually instead of having PL do the
    backward pass."""

    @dataclass
    class HParams(Algorithm.HParams):
        """Hyper-parameters of this example algorithm."""

        lr: float = 0.1

        gradient_noise_std: float = 0.01
        """Standard deviation of the Gaussian noise added to the gradients."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: nn.Module,
        hp: ManualGradientsExample.HParams | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        self.hp = hp or self.HParams()
        # Just to let the type checker know the right type.
        self.hp: ManualGradientsExample.HParams

        # Setting this to False tells PL that we will be calculating the gradients manually.
        # This turns off a few nice things in PL that we might not care about here, such as
        # easy multi-gpu / multi-node / TPU / mixed precision training.
        self.automatic_optimization = False

        # Instantiate any lazy weights with a dummy forward pass (optional).
        self.example_input_array = torch.zeros(
            (datamodule.batch_size, *datamodule.dims), device=self.device
        )
        self.network(self.example_input_array)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int
    ) -> ClassificationOutputs:
        return self.shared_step(batch, batch_index, "train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int
    ) -> ClassificationOutputs:
        return self.shared_step(batch, batch_index, "val")

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int, phase: PhaseStr
    ) -> ClassificationOutputs:
        """Performs a training/validation/test step."""
        x, y = batch
        logits = self(x)

        loss = torch.nn.functional.cross_entropy(logits, y)

        if phase == "train":
            # We don't care about AMP, TPUs, gradient accumulation here, so just get the "real"
            # optimizers instead of the PL optimizer wrappers:
            optimizers = self.optimizers(use_pl_optimizer=False)
            # We only have one optimizer in this example. Otherwise we'd have a list here.
            assert not isinstance(optimizers, list)
            optimizer = optimizers

            optimizer.zero_grad()

            # NOTE: Whenever possible, if you have a simple "loss" tensor, use `manual_backward`.
            # self.manual_backward(loss)
            # However, in this example here, it still works even if we manipulate the grads
            # directly. We're also not training on multiple GPUs, which makes this easier.

            # NOTE: You don't need to call `loss.backward()`, you could also just set .grads
            # directly!
            self.manual_backward(loss)

            for name, parameter in self.named_parameters():
                assert parameter.grad is not None, name
                parameter.grad += self.hp.gradient_noise_std * torch.randn_like(parameter.grad)

            optimizer.step()

        return {"y": y, "logits": logits, "loss": loss.detach()}

    def configure_optimizers(self):
        """Creates the optimizer(s) and learning rate scheduler(s)."""
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)

    def configure_callbacks(self):
        return super().configure_callbacks() + [
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        ]

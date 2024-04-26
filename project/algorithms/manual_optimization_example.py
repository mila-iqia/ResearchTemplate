from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from project.algorithms.bases.image_classification import ImageClassificationAlgorithm
from project.datamodules.bases.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.types import ClassificationOutputs, PhaseStr


class ManualGradientsExample(ImageClassificationAlgorithm):
    """Example of an algorithm that calculates the gradients manually instead of having PL do the
    backward pass."""

    @dataclass
    class HParams(ImageClassificationAlgorithm.HParams):
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
        super().__init__(datamodule=datamodule, network=network, hp=hp or self.HParams())
        # Just to let the type checker know the right type.
        self.hp: ManualGradientsExample.HParams

        # Setting this to False tells PL that we will be calculating the gradients manually.
        # This turns off a few nice things in PL that we might not care about here, such as
        # easy multi-gpu / multi-node / TPU / mixed precision training.
        self.automatic_optimization = False

        # Instantiate any lazy weights with a dummy forward pass (optional).
        self.network(self.example_input_array)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> ClassificationOutputs:
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> ClassificationOutputs:
        return self.shared_step(batch, batch_idx, "val")

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr
    ) -> ClassificationOutputs:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training across GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """
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
            loss.backward()

            for name, parameter in self.named_parameters():
                if parameter.grad is None:
                    continue
                parameter.grad += self.hp.gradient_noise_std * torch.randn_like(parameter.grad)

            optimizer.step()

        return {"y": y, "logits": logits, "loss": loss.detach()}

    def configure_optimizers(self):
        """Creates the optimizer(s) and learning rate scheduler(s)."""
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr)

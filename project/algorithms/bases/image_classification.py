from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Required

import torch
from torch import Tensor

from project.algorithms.bases.algorithm import Algorithm, StepOutputDict
from project.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from project.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.types import PhaseStr
from project.utils.types.protocols import Module

# TODO: Remove this `log` dict, perhaps it's better to use self.log of the pl module instead?


class ClassificationOutputs(StepOutputDict):
    """The dictionary format that is minimally required to be returned from
    `training/val/test_step`."""

    logits: Required[Tensor]
    """The un-normalized logits."""

    y: Required[Tensor]
    """The class labels."""


class ImageClassificationAlgorithm[
    BatchType: tuple[Tensor, Tensor],
    StepOutputType: ClassificationOutputs,
](Algorithm[BatchType, StepOutputType], ABC):
    """Base class for a learning algorithm for image classification.

    This is an extension of the LightningModule class from PyTorch Lightning, with some common
    boilerplate code to keep the algorithm implementations as simple as possible.

    The network can be created separately. This makes it easier to compare different algorithms on the same architecture (e.g. your method vs a baseline).
    """

    @dataclass
    class HParams(Algorithm.HParams):
        """Hyper-parameters of the algorithm."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule[BatchType],
        network: Module[[Tensor], Tensor],
        hp: ImageClassificationAlgorithm.HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.datamodule: ImageClassificationDataModule
        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        # TODO: Check if PL now moves the `example_input_array` to the right device automatically.
        # If possible, we'd like to remove any reference to the device from the algorithm.
        self.example_input_array = torch.zeros(
            [datamodule.batch_size, *datamodule.dims],
            device=self.device,
        )

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int
    ) -> ClassificationOutputs:
        """Performs a training step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int
    ) -> ClassificationOutputs:
        """Performs a validation step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> ClassificationOutputs:
        """Performs a test step."""
        return self.shared_step(batch=batch, batch_index=batch_index, phase="test")

    def predict_step(self, batch: Tensor, batch_index: int, dataloader_idx: int):
        """Performs a prediction step."""
        return self.predict(batch)

    def predict(self, x: Tensor) -> Tensor:
        """Predict the classification labels."""
        assert self.network is not None
        return self.network(x).argmax(-1)

    @abstractmethod
    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_index: int, phase: PhaseStr
    ) -> ClassificationOutputs:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training across GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """

    def training_step_end(self, step_output: ClassificationOutputs) -> ClassificationOutputs:
        """Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        return self.shared_step_end(step_output, phase="train")

    def validation_step_end(self, step_output: ClassificationOutputs) -> ClassificationOutputs:
        return self.shared_step_end(step_output, phase="val")

    def test_step_end(self, step_output: ClassificationOutputs) -> ClassificationOutputs:
        return self.shared_step_end(step_output, phase="test")

    def shared_step_end(
        self, step_output: ClassificationOutputs, phase: PhaseStr
    ) -> ClassificationOutputs:
        fused_output = step_output.copy()
        loss: Tensor | float | None = step_output.get("loss", None)

        if isinstance(loss, Tensor) and loss.shape:
            # Replace the loss with its mean. This is useful when automatic
            # optimization is enabled, for example in the example algo, where each replica
            # returns the un-reduced cross-entropy loss. Here we need to reduce it to a scalar.
            fused_output["loss"] = loss.mean()

        if loss is not None:
            self.log(f"{phase}/loss", torch.as_tensor(loss).mean(), sync_dist=True)

        return fused_output

    def configure_callbacks(self):
        return [
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        ]

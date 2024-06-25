from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import torch.testing
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from project.algorithms.algorithm import Algorithm
from project.algorithms.algorithm_test import (
    AlgorithmTests,
    CheckBatchesAreTheSameAtEachStep,
    MetricShouldImprove,
)
from project.algorithms.callbacks.classification_metrics import ClassificationOutputs
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.types import DataModule
from project.utils.types.protocols import (
    ClassificationDataModule,
)

# Having tuple[torch.Tensor, torch.Tensor] as the batch type isn't ideal.


class ClassificationAlgorithmTests[
    AlgorithmType: Algorithm[tuple[Tensor, Tensor], ClassificationOutputs]
](AlgorithmTests[AlgorithmType]):
    """Test suite for (image) classification algorithms."""

    unsupported_datamodule_types: ClassVar[list[type[DataModule]]] = []
    unsupported_network_types: ClassVar[list[type[nn.Module]]] = []
    _supported_datamodule_types: ClassVar[list[type[ClassificationDataModule]]] = [
        # VisionDataModule,
        ClassificationDataModule,  # type: ignore (we actually support this case).
        # ImageClassificationDataModule,
    ]

    metric_name: ClassVar[str] = "train/accuracy"
    """The main  metric to inspect to check if training is working."""
    lower_is_better: ClassVar[bool] = False

    def test_output_shapes(
        self,
        algorithm: AlgorithmType,
        training_batch: tuple[Tensor, Tensor],
    ):
        """Tests that the output of the algorithm has the correct shape."""
        x, y = training_batch
        output = algorithm(x)
        if isinstance(output, tuple):
            y_pred, x_hat = algorithm(x)
        else:
            y_pred = output
        assert isinstance(y_pred, Tensor)
        assert isinstance(algorithm.datamodule, ClassificationDataModule)
        if y_pred.dtype.is_floating_point:
            # y_pred should be the logits.
            assert y_pred.shape == (y.shape[0], algorithm.datamodule.num_classes)
        else:
            # y_pred might be the sampled classes (e.g. REINFORCE).
            assert y_pred.shape == (y.shape[0],)
            assert y_pred.max() < algorithm.datamodule.num_classes

    @pytest.fixture(scope="class")
    def training_batch(
        self, datamodule: ImageClassificationDataModule, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Returns a batch of data from the training set of the datamodule."""
        datamodule.prepare_data()
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        return (batch[0].to(device), batch[1].to(device))

    @pytest.fixture(scope="class")
    def repeat_first_batch_dataloader(
        self,
        training_batch: tuple[Tensor, Tensor],
        n_updates: int,
    ):
        """Returns a dataloader that yields a exactly the same batch over and over again.

        The image transforms are only executed once here, so the exact same tensors are served over
        and over again. This is different than using `overfit_batches=1` in the trainer and\
        `num_epochs=n_updates`, which would give the same batch but with a (potentially different
        image transformation) every time.
        """
        # Doing this just in case the algorithm wraps the datamodule somehow.
        # dm = getattr(algorithm, "datamodule", datamodule)
        assert len(training_batch)
        dataset = TensorDataset(*training_batch)
        # need `start` to be of the same type, and it's hard to make an empty TensorDataset.
        n_batches_dataset = sum([dataset] * (n_updates - 1), start=dataset)

        batch_size = training_batch[0].shape[0]
        train_dl = DataLoader(n_batches_dataset, batch_size=batch_size, shuffle=False)
        torch.testing.assert_close(next(iter(train_dl)), training_batch)
        return train_dl

    @pytest.mark.slow
    @pytest.mark.timeout(10)
    def test_overfit_exact_same_training_batch(
        self,
        algorithm: AlgorithmType,
        repeat_first_batch_dataloader: DataLoader,
        accelerator: str,
        devices: list[int],
        n_updates: int,
        tmp_path: Path,
    ):
        """Perform `n_updates` training steps on exactly the same batch of training data."""

        testing_callbacks = self.get_testing_callbacks() + [
            CheckBatchesAreTheSameAtEachStep(),
            MetricShouldImprove(metric=self.metric_name, lower_is_better=self.lower_is_better),
        ]
        self._train(
            algorithm=algorithm,
            train_dataloader=repeat_first_batch_dataloader,
            accelerator=accelerator,
            devices=devices,
            max_epochs=1,
            limit_train_batches=n_updates,
            limit_val_batches=0.0,
            limit_test_batches=0.0,
            tmp_path=tmp_path,
            testing_callbacks=testing_callbacks,
        )

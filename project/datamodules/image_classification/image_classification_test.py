from pathlib import Path

import pytest
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture
from torch import Tensor
from torchvision.tv_tensors import Image

from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.vision_test import VisionDataModuleTests


class ImageClassificationDataModuleTests[
    ImageClassificationDataModuleType: ImageClassificationDataModule
](VisionDataModuleTests[ImageClassificationDataModuleType]):
    """Tests for a datamodule/dataset for image classification.

    This is a simple data regression test for now.
    For each of the `train_dataloader`, `valid_dataloader`
    """

    def test_first_batch(  # type: ignore
        self,
        datamodule: ImageClassificationDataModuleType,
        batch: tuple[Image, Tensor],
        request: pytest.FixtureRequest,
        tensor_regression: TensorRegressionFixture,
        original_datadir: Path,
        stage: RunningStage,
        datadir: Path,
    ):
        assert len(batch) == 2
        x, y = batch
        assert y.shape == (x.shape[0],)
        assert (y < datamodule.num_classes).all()
        assert (y >= 0).all()
        super().test_first_batch(
            datamodule=datamodule,
            batch=batch,
            request=request,
            tensor_regression=tensor_regression,
            original_datadir=original_datadir,
            stage=stage,
            datadir=datadir,
        )

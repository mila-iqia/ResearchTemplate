import pytest

from project.conftest import setup_with_overrides
from project.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from project.datamodules.image_classification.imagenet import ImageNetDataModule
from project.utils.testutils import needs_network_dataset_dir


@pytest.mark.slow
@needs_network_dataset_dir("imagenet")
@setup_with_overrides("datamodule=imagenet")
class TestImageNetDataModule(ImageClassificationDataModuleTests[ImageNetDataModule]): ...

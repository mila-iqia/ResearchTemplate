from project.conftest import setup_with_overrides
from project.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from project.datamodules.image_classification.mnist import MNISTDataModule


@setup_with_overrides("datamodule=mnist")
class TestMNISTDataModule(ImageClassificationDataModuleTests[MNISTDataModule]): ...

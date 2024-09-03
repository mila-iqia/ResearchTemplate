"""Datamodules (datasets + preprocessing + dataloading)

See the :ref:`lightning.LightningDataModule` class for more information.
"""

from .image_classification import ImageClassificationDataModule
from .image_classification.cifar10 import CIFAR10DataModule, cifar10_normalization
from .image_classification.fashion_mnist import FashionMNISTDataModule
from .image_classification.imagenet import ImageNetDataModule
from .image_classification.imagenet32 import ImageNet32DataModule, imagenet32_normalization
from .image_classification.inaturalist import INaturalistDataModule
from .image_classification.mnist import MNISTDataModule
from .text.hf_text import HFDataModule
from .vision import VisionDataModule

__all__ = [
    "cifar10_normalization",
    "CIFAR10DataModule",
    "FashionMNISTDataModule",
    "INaturalistDataModule",
    "ImageClassificationDataModule",
    "imagenet32_normalization",
    "ImageNet32DataModule",
    "ImageNetDataModule",
    "MNISTDataModule",
    "VisionDataModule",
    "HFDataModule",
]

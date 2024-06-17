from .image_classification import ImageClassificationDataModule
from .image_classification.cifar10 import CIFAR10DataModule, cifar10_normalization
from .image_classification.fashion_mnist import FashionMNISTDataModule
from .image_classification.imagenet32 import ImageNet32DataModule, imagenet32_normalization
from .image_classification.mnist import MNISTDataModule
from .vision.base import VisionDataModule
from .vision.imagenet import ImageNetDataModule

__all__ = [
    "cifar10_normalization",
    "CIFAR10DataModule",
    "FashionMNISTDataModule",
    "ImageClassificationDataModule",
    "imagenet32_normalization",
    "ImageNet32DataModule",
    "ImageNetDataModule",
    "MNISTDataModule",
    "VisionDataModule",
]

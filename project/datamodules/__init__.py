from .cifar10 import CIFAR10DataModule, cifar10_normalization
from .fashion_mnist import FashionMNISTDataModule
from .image_classification import ImageClassificationDataModule
from .imagenet32 import ImageNet32DataModule, imagenet32_normalization
from .mnist import MNISTDataModule
from .vision import VisionDataModule

__all__ = [
    "CIFAR10DataModule",
    "cifar10_normalization",
    "FashionMNISTDataModule",
    "ImageClassificationDataModule",
    "ImageNet32DataModule",
    "imagenet32_normalization",
    "MNISTDataModule",
    "VisionDataModule",
]

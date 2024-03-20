from .bases.vision import VisionDataModule
from .cifar10 import CIFAR10DataModule, cifar10_normalization
from .fashion_mnist import FashionMNISTDataModule
from .image_classification import ImageClassificationDataModule
from .imagenet32 import ImageNet32DataModule, imagenet32_normalization
from .mnist import MNISTDataModule
from .rl.rl_datamodule import RlDataModule

__all__ = [
    "cifar10_normalization",
    "CIFAR10DataModule",
    "FashionMNISTDataModule",
    "ImageClassificationDataModule",
    "imagenet32_normalization",
    "ImageNet32DataModule",
    "MNISTDataModule",
    "RlDataModule",
    "VisionDataModule",
]

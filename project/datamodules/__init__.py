from .bases.image_classification import ImageClassificationDataModule
from .bases.vision import VisionDataModule
from .rl.rl_datamodule import RlDataModule
from .vision.cifar10 import CIFAR10DataModule, cifar10_normalization
from .vision.fashion_mnist import FashionMNISTDataModule
from .vision.imagenet32 import ImageNet32DataModule, imagenet32_normalization
from .vision.mnist import MNISTDataModule

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

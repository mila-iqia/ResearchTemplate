from __future__ import annotations

from torch import Tensor
from torchvision.tv_tensors import Image

from project.datamodules.vision import VisionDataModule
from project.utils.types import C, H, W


class ImageClassificationDataModule[BatchType: tuple[Image, Tensor]](VisionDataModule[BatchType]):
    """Lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

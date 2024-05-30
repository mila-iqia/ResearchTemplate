from __future__ import annotations

from torch import Tensor

from project.datamodules.vision.base import VisionDataModule
from project.utils.types import C, H, W


class ImageClassificationDataModule[BatchType: tuple[Tensor, Tensor]](VisionDataModule[BatchType]):
    """Protocol that describes lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

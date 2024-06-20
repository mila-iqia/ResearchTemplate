from __future__ import annotations

from torch import Tensor

from project.datamodules.vision import VisionDataModule
from project.utils.types import C, H, W

# todo: decide if this should be a protocol or an actual base class (currently a base class).


class ImageClassificationDataModule[BatchType: tuple[Tensor, Tensor]](VisionDataModule[BatchType]):
    """Lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

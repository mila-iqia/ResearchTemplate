from __future__ import annotations

from typing import ClassVar, TypeVar

from torch import Tensor
from torchvision.tv_tensors import Image

from project.datamodules.vision import VisionDataModule
from project.utils.typing_utils import C, H, W
from project.utils.typing_utils.protocols import ClassificationDataModule

ImageBatchType = TypeVar("ImageBatchType", bound=tuple[Image, Tensor])


class ImageClassificationDataModule(
    VisionDataModule[ImageBatchType], ClassificationDataModule[ImageBatchType]
):
    """Lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: ClassVar[tuple[C, H, W]]
    """A tuple describing the shape of the data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

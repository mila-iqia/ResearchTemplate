from __future__ import annotations

from torch import Tensor
from torchvision.tv_tensors import Image
from typing_extensions import TypeVar

from project.datamodules.vision import VisionDataModule
from project.utils.typing_utils import C, H, W
from project.utils.typing_utils.protocols import ClassificationDataModule

ImageBatchType = TypeVar(
    "ImageBatchType", bound=tuple[Image, Tensor], default=tuple[Image, Tensor]
)


# todo: this should probably be a protocol. The only issue with that is that we do `issubclass` in
# tests to determine which datamodule configs are for image classification, so we can't do that
# with a Protocol.


class ImageClassificationDataModule(
    VisionDataModule[ImageBatchType], ClassificationDataModule[ImageBatchType]
):
    """Lightning data modules for image classification."""

    # This just adds the `num_classes` property to `VisionDataModule`.

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

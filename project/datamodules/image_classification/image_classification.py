from __future__ import annotations

from typing import ClassVar, TypeVar

from torch import Tensor
from torchvision import transforms
from torchvision.tv_tensors import Image

from project.datamodules.vision import VisionDataModule
from project.utils.typing_utils import C, H, W
from project.utils.typing_utils.protocols import ClassificationDataModule
from project.utils.utils import logger

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
        if not self.normalize:
            remove_normalization_from_transforms(self)


def remove_normalization_from_transforms(
    datamodule: ImageClassificationDataModule,
) -> None:
    transform_properties = (
        datamodule.train_transforms,
        datamodule.val_transforms,
        datamodule.test_transforms,
    )
    for transform_list in transform_properties:
        if transform_list is None:
            continue
        assert isinstance(transform_list, transforms.Compose)
        if isinstance(transform_list.transforms[-1], transforms.Normalize):
            t = transform_list.transforms.pop(-1)
            logger.info(f"Removed normalization transform {t} since datamodule.normalize=False")
        if any(isinstance(t, transforms.Normalize) for t in transform_list.transforms):
            raise RuntimeError(
                f"Unable to remove all the normalization transforms from datamodule {datamodule}: "
                f"{transform_list}"
            )

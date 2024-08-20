from __future__ import annotations

from typing import TypeVar

from torch import Tensor
from torchvision.tv_tensors import Image

from project.datamodules.vision import VisionDataModule
from project.utils.typing_utils import C, H, W
from project.utils.typing_utils.protocols import ClassificationDataModule

# todo: need to decide whether this should be a base class or just a protocol.
# - IF this is a protocol, then we can't use issubclass with it, so it can't be used in the
# `supported_datamodule_types` field on AlgorithmTests subclasses (for example `ClassificationAlgorithmTests`).
BatchType = TypeVar("BatchType", bound=tuple[Image, Tensor])


class ImageClassificationDataModule(
    VisionDataModule[BatchType], ClassificationDataModule[BatchType]
):
    """Lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

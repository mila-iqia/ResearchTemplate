from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from project.utils.types import C, H, StageStr, W
from project.utils.types.protocols import DataModule


@runtime_checkable
class ImageClassificationDataModule[BatchType: tuple[Tensor, Tensor]](
    DataModule[BatchType], Protocol
):
    """Protocol that describes lightning data modules for image classification."""

    num_classes: int
    """Number of classes in the dataset."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

    name: str = ""
    """Datamodule name."""

    dataset_cls: type[Dataset]
    """Dataset class to use."""

    data_dir: str | None = None
    val_split: int | float = 0.2
    num_workers: int = 16
    normalize: bool = False
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False

    train_transforms: Callable[..., Any] | None
    val_transforms: Callable[..., Any] | None
    test_transforms: Callable[..., Any] | None

    def prepare_data(self): ...

    def setup(self, stage: StageStr | None = None): ...

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader[tuple[Tensor, Tensor]]: ...

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader[tuple[Tensor, Tensor]]: ...

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader[tuple[Tensor, Tensor]]: ...

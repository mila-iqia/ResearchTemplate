from __future__ import annotations
from typing import Iterable, Protocol, runtime_checkable
from typing import TypeVar

from project.utils.types import StageStr

BatchType_co = TypeVar("BatchType_co", covariant=True)


@runtime_checkable
class DataModule(Protocol[BatchType_co]):
    """Protocol that shows the expected attributes / methods of the `LightningDataModule` class.

    This is used to type hint the batches that are yielded by the DataLoaders.
    """

    # batch_size: int

    def prepare_data(self) -> None:
        ...

    def setup(self, stage: StageStr) -> None:
        ...

    def train_dataloader(self) -> Iterable[BatchType_co]:
        ...

    # Optional:
    # def val_dataloader(self) -> Iterable[BatchType_co] | None:
    #     return None

    # def test_dataloader(self) -> Iterable[BatchType_co] | None:
    #     return None

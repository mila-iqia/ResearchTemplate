from __future__ import annotations

import inspect
import os
from abc import abstractmethod
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, ClassVar

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from typing_extensions import Concatenate, ParamSpec

from project.utils.types import C, H, W

from .datamodule import DataModule

P = ParamSpec("P")

SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
logger = get_logger(__name__)


class VisionDataModule[BatchType_co](LightningDataModule, DataModule[BatchType_co]):
    """A LightningDataModule for image datasets."""

    name: ClassVar[str] = ""
    """Dataset name."""

    dataset_cls: ClassVar[type[VisionDataset]]
    """Dataset class to use."""

    dims: ClassVar[tuple[C, H, W]]
    """A tuple describing the shape of the data."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        val_split: int | float = 0.2,
        num_workers: int | None = None,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            train_transforms: transformations you can apply to train dataset
            val_transforms: transformations you can apply to validation dataset
            test_transforms: transformations you can apply to test dataset
        """

        super().__init__()

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        if num_workers is None:
            num_workers = num_cpus_on_node()
            logger.debug(f"Setting the number of dataloader workers to {num_workers}.")
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.EXTRA_ARGS = kwargs

        self.train_kwargs = self.EXTRA_ARGS.copy()
        self.test_kwargs = self.EXTRA_ARGS.copy()
        if _has_constructor_argument(self.dataset_cls, "train"):
            self.train_kwargs["train"] = True
            self.test_kwargs["train"] = False

        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.train_dl_rng_seed = int(
            torch.randint(0, int(1e6), (1,), generator=self._rng).item()
        )
        self.val_dl_rng_seed = int(
            torch.randint(0, int(1e6), (1,), generator=self._rng).item()
        )
        self.test_dl_rng_seed = int(
            torch.randint(0, int(1e6), (1,), generator=self._rng).item()
        )

        self.dataset_test: VisionDataset | None = None

    @property
    def train_transforms(self) -> Callable[..., Any] | None:
        """Optional transforms (or collection of transforms) you can apply to train dataset."""
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t: Callable) -> None:
        self._train_transforms = t

    @property
    def val_transforms(self) -> Callable[..., Any] | None:
        """Optional transforms (or collection of transforms) you can apply to validation
        dataset."""
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t: Callable) -> None:
        self._val_transforms = t

    @property
    def test_transforms(self) -> Callable[..., Any] | None:
        """Optional transforms (or collection of transforms) you can apply to test dataset."""
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t: Callable) -> None:
        self._test_transforms = t

    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        # Call with `train=True` and `train=False` if there is such an argument.

        train_kwargs = self.train_kwargs.copy()
        test_kwargs = self.test_kwargs.copy()
        if _has_constructor_argument(self.dataset_cls, "download"):
            train_kwargs["download"] = True
            test_kwargs["download"] = True
        logger.info(
            f"Preparing {self.name} dataset training split in {self.data_dir} with {train_kwargs}"
        )
        self.dataset_cls(str(self.data_dir), **train_kwargs)
        if test_kwargs != train_kwargs:
            logger.info(
                f"Preparing {self.name} dataset test spit in {self.data_dir} with {test_kwargs=}"
            )
            self.dataset_cls(str(self.data_dir), **test_kwargs)

    def setup(self, stage: str | None = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            val_transforms = (
                self.default_transforms()
                if self.val_transforms is None
                else self.val_transforms
            )

            dataset_train = self.dataset_cls(
                str(self.data_dir),
                transform=train_transforms,
                **self.train_kwargs,
            )
            dataset_val = self.dataset_cls(
                str(self.data_dir),
                transform=val_transforms,
                **self.train_kwargs,  # todo: Assuming those are the same for now.
            )

            # Split
            self.dataset_train = self._split_dataset(dataset_train, train=True)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = (
                self.default_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )
            self.dataset_test = self.dataset_cls(
                str(self.data_dir), transform=test_transforms, **self.test_kwargs
            )

    def _split_dataset(self, dataset: VisionDataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> list[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    @abstractmethod
    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""

    def train_dataloader(
        self,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(
            self.dataset_train,
            _dataloader_fn=_dataloader_fn,
            *args,
            **(
                dict(
                    shuffle=self.shuffle,
                    generator=torch.Generator().manual_seed(self.train_dl_rng_seed),
                )
                | kwargs
            ),
        )

    def val_dataloader(
        self,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataLoader:
        """The val dataloader."""

        return self._data_loader(
            self.dataset_val,
            _dataloader_fn=_dataloader_fn,
            *args,
            **(
                dict(generator=torch.Generator().manual_seed(self.val_dl_rng_seed))
                | kwargs
            ),
        )

    def test_dataloader(
        self,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataLoader:
        """The test dataloader."""
        if self.dataset_test is None:
            self.setup("test")
        assert self.dataset_test is not None
        return self._data_loader(
            self.dataset_test,
            _dataloader_fn=_dataloader_fn,
            *args,
            **(
                dict(generator=torch.Generator().manual_seed(self.test_dl_rng_seed))
                | kwargs
            ),
        )

    def _data_loader(
        self,
        dataset: Dataset,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *dataloader_args: P.args,
        **dataloader_kwargs: P.kwargs,
    ) -> DataLoader:
        dataloader_kwargs = (
            dict(  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
            )
            | dataloader_kwargs
        )
        return _dataloader_fn(dataset, *dataloader_args, **dataloader_kwargs)


def _has_constructor_argument(cls: type[VisionDataset], arg: str) -> bool:
    # TODO: Would be more accurate to check if cls has either download or a **kwargs argument and
    # then check if the base class constructor takes a `download` argument.
    sig = inspect.signature(cls.__init__)
    # Check if sig has a **kwargs argument
    if arg in sig.parameters:
        return True
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return _has_constructor_argument(cls.__base__, arg)
    return False


def num_cpus_on_node() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()

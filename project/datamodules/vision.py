from __future__ import annotations

import inspect
import os
from abc import abstractmethod
from collections.abc import Callable
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar, Concatenate, Literal, ParamSpec, TypeVar

import torch
import torchvision.transforms
import torchvision.transforms.v2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import collate_tensor_fn, default_collate_fn_map
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image, set_return_type

from project.utils.env_vars import DATA_DIR, NUM_WORKERS
from project.utils.typing_utils import C, H, W
from project.utils.typing_utils.protocols import DataModule

logger = get_logger(__name__)

BatchType_co = TypeVar("BatchType_co", covariant=True)
P = ParamSpec("P")


class VisionDataModule(LightningDataModule, DataModule[BatchType_co]):
    """A LightningDataModule for image datasets.

    (Taken from pl_bolts which is not very well maintained.)
    """

    name: str | None = ""
    """Dataset name."""

    dataset_cls: ClassVar[type[VisionDataset]]
    """Dataset class to use."""

    dims: tuple[C, H, W]
    """A tuple describing the shape of the data."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        val_split: int | float = 0.2,
        num_workers: int = NUM_WORKERS,
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
        self.data_dir: Path = Path(data_dir or DATA_DIR)
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transforms = train_transforms or self.default_transforms()
        self.val_transforms = val_transforms or torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.test_transforms = test_transforms or torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            ]
        )
        if (
            not normalize
            and train_transforms is not None
            and _contains_normalization_transform(train_transforms)
        ):
            logger.warning(
                "You passed `normalize=False` but `train_transforms` contains a normalization transform. "
                "The provided normalization transform will be applied."
            )

        # todo: what about the shuffling at each epoch?
        _rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.train_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())
        self.val_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())
        self.test_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())

        self.test_dataset_cls = self.dataset_cls

        self.dataset_train: Dataset | None = None
        self.dataset_val: Dataset | None = None
        self.dataset_test: VisionDataset | None = None

        self.EXTRA_ARGS = kwargs
        self.train_kwargs = self.EXTRA_ARGS | {"transform": self.train_transforms}
        self.valid_kwargs = self.EXTRA_ARGS | {"transform": self.val_transforms}
        self.test_kwargs = self.EXTRA_ARGS | {"transform": self.test_transforms}

        if _has_constructor_argument(self.dataset_cls, "train"):
            self.train_kwargs["train"] = True
            self.valid_kwargs["train"] = True
            self.test_kwargs["train"] = False

        self.batch_size_per_device: int = batch_size
        self.save_hyperparameters(
            logger=False, ignore=["train_transforms", "val_transforms", "test_transforms"]
        )

    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        # Call with `train=True` and `train=False` if there is such an argument.

        train_kwargs = self.train_kwargs.copy()
        test_kwargs = self.test_kwargs.copy()
        if _has_constructor_argument(self.dataset_cls, "download"):
            train_kwargs["download"] = True
            test_kwargs["download"] = True
        logger.debug(
            f"Preparing {self.name} dataset training split in {self.data_dir} with {train_kwargs}"
        )
        self.dataset_cls(str(self.data_dir), **train_kwargs)
        if test_kwargs != train_kwargs:
            logger.debug(
                f"Preparing {self.name} dataset test spit in {self.data_dir} with {test_kwargs=}"
            )
            self.test_dataset_cls(str(self.data_dir), **test_kwargs)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        if stage in ["fit", "validate"] or stage is None:
            logger.debug(f"creating training dataset with kwargs {self.train_kwargs}")
            dataset_train = self.dataset_cls(
                str(self.data_dir),
                **self.train_kwargs,
            )
            logger.debug(f"creating validation dataset with kwargs {self.valid_kwargs}")
            dataset_val = self.dataset_cls(
                str(self.data_dir),
                **self.valid_kwargs,
            )

            # TODO: If we support more datasets than image classification, we could add this:
            # dataset_train = wrap_dataset_for_transforms_v2(dataset_train)
            # dataset_val = wrap_dataset_for_transforms_v2(dataset_val)

            # Train/validation split.
            # NOTE: the dataset is created twice (with the right transforms) and split in the same
            # way, such that there is no overlap in indices between train and validation sets.
            self.dataset_train = self._split_dataset(dataset_train, train=True)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            logger.debug(f"creating test dataset with kwargs {self.train_kwargs}")
            self.dataset_test = self.test_dataset_cls(str(self.data_dir), **self.test_kwargs)

        # Divide batch size by the number of devices.
        # Taken from https://github.com/ashleve/lightning-hydra-template/commit/1fb540580602e73ba6481ab5e4e86187690d2dbf
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

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
        assert self.dataset_train is not None
        kwargs = kwargs.copy()  # type: ignore
        kwargs.setdefault("shuffle", self.shuffle)
        kwargs.setdefault("generator", torch.Generator().manual_seed(self.train_dl_rng_seed))
        return self._data_loader(
            self.dataset_train,
            _dataloader_fn=_dataloader_fn,
            *args,
            **kwargs,
        )

    def val_dataloader(
        self,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataLoader:
        """The val dataloader."""
        assert self.dataset_val is not None
        kwargs = kwargs.copy()  # type: ignore
        kwargs.setdefault("generator", torch.Generator().manual_seed(self.val_dl_rng_seed))
        return self._data_loader(
            self.dataset_val,
            _dataloader_fn=_dataloader_fn,
            *args,
            **kwargs,
        )

    def test_dataloader(
        self,
        _dataloader_fn: Callable[Concatenate[Dataset, P], DataLoader] = DataLoader,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataLoader:
        """The test dataloader."""
        assert self.dataset_test is not None
        kwargs = kwargs.copy()  # type: ignore
        kwargs.setdefault("generator", torch.Generator().manual_seed(self.test_dl_rng_seed))
        return self._data_loader(
            self.dataset_test,
            _dataloader_fn=_dataloader_fn,
            *args,
            **kwargs,
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
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )
            | dataloader_kwargs
        )

        return _dataloader_fn(dataset, *dataloader_args, **dataloader_kwargs)


def collate_images(
    images: list[Image],
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    with set_return_type("TVTensor"):
        # note: Image is a subclass of Tensor, but list[Image] is not a subclass of list[Tensor]
        image_batch: Image = collate_tensor_fn(images)  # type: ignore
    assert isinstance(image_batch, Image), type(image_batch)

    if image_batch.ndim <= 4:
        # We wouldn't want to return an Image for higher-dimensions, it probably wouldn't make sense.
        # log_once(
        #     message="Collating images into `torchvision.tv_tensors.Image`s", level=logging.INFO
        # )
        return image_batch
    return image_batch.as_subclass(torch.Tensor)


default_collate_fn_map[Image] = collate_images


def _has_constructor_argument(cls: type[VisionDataset], arg: str) -> bool:
    # TODO: Would be more accurate to check if cls has either download or a **kwargs argument and
    # then check if the base class constructor takes a `download` argument.
    sig = inspect.signature(cls.__init__)
    # Check if sig has a **kwargs argument
    if arg in sig.parameters:
        return True
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()) and cls.__base__ is not None:
        return _has_constructor_argument(cls.__base__, arg)
    return False


def num_cpus_on_node() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def _contains_normalization_transform(transforms: Callable) -> bool:
    if isinstance(
        transforms, torchvision.transforms.Normalize | torchvision.transforms.v2.Normalize
    ):
        return True
    if isinstance(transforms, torchvision.transforms.Compose | torchvision.transforms.v2.Compose):
        return any(_contains_normalization_transform(t) for t in transforms.transforms)
    if isinstance(transforms, torch.nn.Sequential):
        return any(_contains_normalization_transform(t) for t in transforms.transforms)
    return False

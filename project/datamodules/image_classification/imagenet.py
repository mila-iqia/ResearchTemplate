"""ImageNet datamodule adapted to the Mila cluster.

Can be used either with a PyTorch-Lightning Trainer, or by itself to easily get efficient
dataloaders for the ImageNet dataset.

Requirements (these are the versions I'm using, but this can probably be loosened a bit).
- pytorch-lightning==1.6.0
- lightning-bolts==0.5
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from multiprocessing import cpu_count
from typing import Literal, NewType

import torchvision.datasets as tvd
from lightning import Trainer

# TODO: reimprot stuff from pl_bolts's ImageNet datamodule.
from pl_bolts.datasets import UnlabeledImagenet  # noqa

# from mila_datamodules.clusters import CURRENT_CLUSTER
# from mila_datamodules.vision.datasets import AdaptedDataset, ImageNet
# from pl_bolts.datamodules.imagenet_datamodule import (
#     ImagenetDataModule as _ImagenetDataModule,
# )
# from pl_bolts.datasets import UnlabeledImagenet
# from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from project.datamodules.image_classification.base import ImageClassificationDataModule
from project.datamodules.image_classification.inaturalist import get_slurm_tmpdir

Stage = Literal["fit", "validate", "test", "predict"]

C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
CURRENT_CLUSTER = os.environ.get("SLURM_CLUSTER")


class ImagenetDataModule(ImageClassificationDataModule):
    """Imagenet DataModule adapted to the Mila cluster.

    - Copies/Extracts the datasets to the `$SLURM_TMPDIR/data/ImageNet` directory.
    - Uses the right number of workers, depending on the SLURM configuration.

    TODO: Unclear why this couldn't just be a VisionDataModule using the ImageNet class.
    TODO: Unclear why this `UnlabeledImageNet` dataset class is needed.
    """

    dataset_cls: type[tvd.ImageNet] = ImageNet

    def __init__(
        self,
        data_dir: str | None = None,
        meta_dir: str | None = None,
        num_imgs_per_val_class: int = 50,
        image_size: int = 224,
        num_workers: int | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        train_transforms: Callable | nn.Module | None = None,
        val_transforms: Callable | nn.Module | None = None,
        test_transforms: Callable | nn.Module | None = None,
    ) -> None:
        if os.environ.get("SLURM_CLUSTER") == "mila":
            slurm_tmpdir = get_slurm_tmpdir()
            assert slurm_tmpdir
            fixed_data_dir = str(slurm_tmpdir / "data" / "ImageNet")
            if data_dir is not None and data_dir != fixed_data_dir:
                warnings.warn(
                    RuntimeWarning(
                        f"Ignoring passed data_dir ({data_dir}), using {fixed_data_dir} instead."
                    )
                )
            data_dir = fixed_data_dir
        else:
            # Not on a SLURM cluster. `data_dir` must be provided (same as the base class).
            # leave it as None, so the base class can raise the error.
            pass
        super().__init__(
            data_dir=data_dir,  # type: ignore
            meta_dir=meta_dir,
            num_imgs_per_val_class=num_imgs_per_val_class,
            image_size=image_size,
            num_workers=num_workers or num_cpus_to_use(),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.persistent_workers = persistent_workers
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.trainer: Trainer | None = None

        # NOTE: Do we want to store the dataset instances on the datamodule? Perhaps it could be
        # useful for restoring state later or something?
        # self.dataset_fit: UnlabeledImagenet | None = None
        # self.dataset_validate: UnlabeledImagenet | None = None
        # self.dataset_test: UnlabeledImagenet | None = None
        # self.dataset_predict: UnlabeledImagenet | None = None

    def prepare_data(self) -> None:
        """Prepares the data, copying the dataset to the SLURM temporary directory.

        NOTE: When using this datamodule without the PyTorch-Lightning Trainer, make sure to call
        prepare_data() before calling train/val/test_dataloader().
        """
        if CURRENT_CLUSTER is not None:
            # Create the dataset.
            self.dataset_cls(split="train")
            self.dataset_cls(split="val")
        super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        # TODO: Use persistent_workers = True kwarg to DataLoader when num_workers > 0 and when
        # in ddp_spawn mode.
        from lightning.pytorch.strategies.ddp_spawn import DDPSpawnStrategy

        if self.trainer and isinstance(self.trainer.strategy, DDPSpawnStrategy):
            # Use `persistent_workers=True`
            # NOTE: Unfortunate that we have to copy all this code from the base class ;(
            transforms = (
                self.train_transform() if self.train_transforms is None else self.train_transforms
            )
            dataset = UnlabeledImagenet(
                self.data_dir,
                num_imgs_per_class=-1,
                num_imgs_per_class_val_split=self.num_imgs_per_val_class,
                meta_dir=self.meta_dir,
                split="train",
                transform=transforms,
            )
            loader: DataLoader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            return loader
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()

    @property
    def train_transforms(self) -> nn.Module | Callable | None:
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, value: nn.Module | Callable | None):
        self._train_transforms = value

    @property
    def val_transforms(self) -> nn.Module | Callable | None:
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, value: nn.Module | Callable | None):
        self._val_transforms = value

    @property
    def test_transforms(self) -> nn.Module | Callable | None:
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, value: nn.Module | Callable | None):
        self._test_transforms = value

    @property
    def dims(self) -> tuple[C, H, W]:
        """A tuple describing the shape of your data. Extra functionality exposed in ``size``.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """
        return self._dims

    @dims.setter
    def dims(self, v: tuple[C, H, W]):
        self._dims = v


def num_cpus_to_use() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    return cpu_count()

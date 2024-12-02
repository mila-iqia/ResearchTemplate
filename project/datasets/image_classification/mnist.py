from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import v2 as transforms

from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.env_vars import DATA_DIR
from project.utils.typing_utils import C, H, W


def mnist_train_transforms():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=28, padding=4, padding_mode="edge"),
            transforms.ToDtype(torch.float32, scale=True),
            mnist_normalization(),
        ]
    )


def mnist_normalization():
    # NOTE: Taken from https://stackoverflow.com/a/67233938/6388696
    # return transforms.Normalize(mean=0.5, std=0.5)
    return transforms.Normalize(mean=[0.1307], std=[0.3081])


def mnist_unnormalization(x: Tensor) -> Tensor:
    # NOTE: Taken from https://stackoverflow.com/a/67233938/6388696
    # return transforms.Normalize(mean=0.5, std=0.5)
    mean = 0.1307
    std = 0.3081
    return (x * std) + mean


class MNISTDataModule(ImageClassificationDataModule):
    """
    .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
        :width: 400
        :alt: MNIST

    Specs:
        - 10 classes (1 per digit)
        - Each image is (1 x 28 x 28)

    Standard MNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import MNISTDataModule

        dm = MNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = "mnist"
    dataset_cls = MNIST
    dims = (C(1), H(28), W(28))
    num_classes = 10

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        val_split: int | float = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
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
        """
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    def default_transforms(self) -> Callable:
        if self.normalize:
            return transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    mnist_normalization(),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

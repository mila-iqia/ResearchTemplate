from __future__ import annotations

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 as transforms

from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.typing_utils import C, H, W


def cifar10_train_transforms():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=32, padding=4, padding_mode="edge"),
            transforms.ToDtype(torch.float32, scale=True),
            cifar10_normalization(),
            transforms.ToImage(),
        ]
    )


def cifar10_normalization() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )


def cifar10_unnormalization(x: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor([x / 255.0 for x in [125.3, 123.0, 113.9]], device=x.device).view(
        [1, 1, 3]
    )
    std = torch.as_tensor([x / 255.0 for x in [63.0, 62.1, 66.7]], device=x.device).view([1, 1, 3])
    assert x.shape[-3:] == (32, 32, 3), x.shape
    return (x * std) + mean


class CIFAR10DataModule(ImageClassificationDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-10

    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR10, train, val, test splits and transforms

    Transforms::

        transforms = transform_lib.Compose([
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.float32, scale=True),
            transform_lib.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar10"
    dataset_cls = CIFAR10
    dims = (C(3), H(32), W(32))
    num_classes = 10

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    def default_transforms(self) -> transforms.Compose:
        if self.normalize:
            return transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    cifar10_normalization(),
                    transforms.ToImage(),  # unsure if this is necessary.
                ]
            )
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

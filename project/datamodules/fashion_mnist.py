from __future__ import annotations
from typing import Any, Callable
from torchvision import transforms as transform_lib
from torchvision.datasets import FashionMNIST
from project.utils.types import C, H, W
from project.datamodules.vision import VisionDataModule


class FashionMNISTDataModule(VisionDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/
        wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset.png
        :width: 400
        :alt: Fashion MNIST

    Specs:
        - 10 classes (1 per type)
        - Each image is (1 x 28 x 28)

    Standard FashionMNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import FashionMNISTDataModule

        dm = FashionMNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = "fashion_mnist"
    dataset_cls = FashionMNIST
    dims = (C(1), H(28), W(28))

    def __init__(
        self,
        data_dir: str | None = None,
        val_split: int | float = 0.2,
        num_workers: int | None = 0,
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
            data_dir: Root directory of dataset.
            val_split: Percent (float) or number (int) of samples to use for the validation split.
            num_workers: Number of workers to use for loading data.
            normalize: If ``True``, applies image normalization.
            batch_size: Number of samples per batch to load.
            seed: Random seed to be used for train/val/test splits.
            shuffle: If ``True``, shuffles the train data every epoch.
            pin_memory: If ``True``, the data loader will copy Tensors into CUDA pinned memory \
                before returning them.
            drop_last: If ``True``, drops the last incomplete batch.
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

    @property
    def num_classes(self) -> int:
        """Returns the number of classes."""
        return 10

    def default_transforms(self) -> Callable:
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return mnist_transforms

from __future__ import annotations

import logging
import math
import os
import shutil
import tarfile
import time
from collections import defaultdict
from collections.abc import Callable
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar, Literal, NewType

import rich
import rich.logging
import torch
import torch.utils.data
import torchvision
import tqdm
from torchvision.datasets import ImageNet
from torchvision.models.resnet import ResNet152_Weights
from torchvision.transforms import v2 as transforms

from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.env_vars import DATA_DIR, NETWORK_DIR, NUM_WORKERS
from project.utils.typing_utils import C, H, W

logger = get_logger(__name__)


def imagenet_normalization():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


ClassIndex = NewType("ClassIndex", int)
ImageIndex = NewType("ImageIndex", int)


class ImageNetDataModule(ImageClassificationDataModule):
    """ImageNet datamodule.

    Extracted from https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datamodules/imagenet_datamodule.py
    - Made this a subclass of VisionDataModule

    Notes:

    - train_dataloader uses the train split of imagenet2012 and puts away a portion of it for the validation split.
    - val_dataloader uses the part of the train split of imagenet2012  that was not used for training via
        `num_imgs_per_val_class`
    - test_dataloader uses the validation split of imagenet2012 for testing.
        - TODO: need to pass num_imgs_per_class=-1 for test dataset and split="test".
    """

    name: str | None = "imagenet"
    """Dataset name."""

    dataset_cls: ClassVar[type[torchvision.datasets.VisionDataset]] = ImageNet
    """Dataset class to use."""

    dims: tuple[C, H, W] = (C(3), H(224), W(224))
    """A tuple describing the shape of the data."""

    num_classes: int = 1000

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        *,
        val_split: int | float = 0.01,
        num_workers: int = NUM_WORKERS,
        normalize: bool = False,
        image_size: int = 224,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        **kwargs,
    ):
        """Creates an ImageNet datamodule (doesn't load or prepare the dataset yet).

        Parameters:
            data_dir: path to the imagenet dataset file
            val_split: save `val_split`% of the training data *of each class* for validation.
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before \
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        self.image_size = image_size
        super().__init__(
            data_dir,
            num_workers=num_workers,
            val_split=val_split,
            shuffle=shuffle,
            pin_memory=pin_memory,
            normalize=normalize,
            seed=seed,
            batch_size=batch_size,
            drop_last=drop_last,
            train_transforms=train_transforms or self.train_transform(),
            val_transforms=val_transforms or self.val_transform(),
            test_transforms=test_transforms or self.test_transform(),
            **kwargs,
        )
        self.dims = (C(3), H(self.image_size), W(self.image_size))
        self.train_kwargs = self.train_kwargs | {"split": "train"}
        self.valid_kwargs = self.valid_kwargs | {"split": "train"}
        self.test_kwargs = self.test_kwargs | {"split": "val"}
        # self.test_dataset_cls = UnlabeledImagenet

    def prepare_data(self) -> None:
        if (
            not NETWORK_DIR
            or not (network_imagenet_dir := NETWORK_DIR / "datasets" / "imagenet").exists()
        ):
            raise NotImplementedError(
                "Assuming that the imagenet dataset can be found at "
                "${NETWORK_DIR:-/network}/datasets/imagenet, (using $NETWORK_DIR if set, else "
                "'/network'), but this path doesn't exist!"
            )

        logger.debug(f"Preparing ImageNet train split in {self.data_dir}...")
        prepare_imagenet(
            self.data_dir,
            network_imagenet_dir=network_imagenet_dir,
            split="train",
        )
        logger.debug(f"Preparing ImageNet val (test) split in {self.data_dir}...")
        prepare_imagenet(
            self.data_dir,
            network_imagenet_dir=network_imagenet_dir,
            split="val",
        )

        super().prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        logger.debug(f"Setup ImageNet datamodule for {stage=}")
        super().setup(stage)

    def _split_dataset(
        self, dataset: torchvision.datasets.VisionDataset, train: bool = True
    ) -> torch.utils.data.Dataset:
        assert isinstance(dataset, ImageNet)
        class_item_indices: dict[ClassIndex, list[ImageIndex]] = defaultdict(list)
        for dataset_index, y in enumerate(dataset.targets):
            class_item_indices[ClassIndex(y)].append(ImageIndex(dataset_index))

        train_val_split_seed = self.seed
        gen = torch.Generator().manual_seed(train_val_split_seed)

        train_class_indices: dict[ClassIndex, list[ImageIndex]] = {}
        valid_class_indices: dict[ClassIndex, list[ImageIndex]] = {}

        for label, dataset_indices in class_item_indices.items():
            num_images_in_class = len(dataset_indices)
            num_valid = math.ceil(self.val_split * num_images_in_class)
            num_train = num_images_in_class - num_valid

            permutation = torch.randperm(len(dataset_indices), generator=gen)
            dataset_indices = torch.tensor(dataset_indices)[permutation].tolist()

            train_indices = dataset_indices[:num_train]
            valid_indices = dataset_indices[num_train:]

            train_class_indices[label] = train_indices
            valid_class_indices[label] = valid_indices

        all_train_indices = sum(train_class_indices.values(), [])
        all_valid_indices = sum(valid_class_indices.values(), [])
        train_dataset = torch.utils.data.Subset(dataset, all_train_indices)
        valid_dataset = torch.utils.data.Subset(dataset, all_valid_indices)
        if train:
            return train_dataset
        return valid_dataset

    def _verify_splits(self, data_dir: str | Path, split: str) -> None:
        dirs = os.listdir(data_dir)
        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def default_transforms(self) -> torch.nn.Module:
        return ResNet152_Weights.IMAGENET1K_V1.transforms

    def train_transform(self) -> torch.nn.Module:
        """The standard imagenet transforms.

        ```python
        transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        ```
        """
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                imagenet_normalization(),
            ]
        )

    def val_transform(self) -> transforms.Compose:
        """The standard imagenet transforms for validation.

        .. code-block:: python

            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        return transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                imagenet_normalization(),
            ]
        )

    # todo: what should be the default transformations for the test set? Same as validation, right?
    test_transform = val_transform


def prepare_imagenet(
    root: Path,
    *,
    split: Literal["train", "val"] = "train",
    network_imagenet_dir: Path,
) -> None:
    """Custom preparation function for ImageNet, using @obilaniu's tar magic in Python form.

    The core of this is equivalent to these bash commands:

    ```bash
    mkdir -p $SLURM_TMPDIR/imagenet/val
    cd       $SLURM_TMPDIR/imagenet/val
    tar  -xf /network/scratch/b/bilaniuo/ILSVRC2012_img_val.tar
    mkdir -p $SLURM_TMPDIR/imagenet/train
    cd       $SLURM_TMPDIR/imagenet/train
    tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar \
         --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
    ```
    """
    if not network_imagenet_dir.exists():
        raise NotImplementedError(
            f"Assuming that we're running on a cluster where {network_imagenet_dir} exists for now."
        )
    val_archive_file_name = "ILSVRC2012_img_val.tar"
    train_archive_file_name = "ILSVRC2012_img_train.tar"
    devkit_file_name = "ILSVRC2012_devkit_t12.tar.gz"
    md5sums_file_name = "md5sums"
    if not root.exists():
        root.mkdir(parents=True)

    def _symlink_if_needed(filename: str, network_imagenet_dir: Path):
        if not (symlink := root / filename).exists():
            symlink.symlink_to(network_imagenet_dir / filename)

    # Create a symlink to the archive in $SLURM_TMPDIR, because torchvision expects it to be
    # there.
    _symlink_if_needed(train_archive_file_name, network_imagenet_dir)
    _symlink_if_needed(val_archive_file_name, network_imagenet_dir)
    _symlink_if_needed(devkit_file_name, network_imagenet_dir)
    # TODO: COPY the file, not symlink it! (otherwise we get some "Read-only filesystem" errors
    # when calling tvd.ImageNet(...). (Probably because the constructor tries to open the file)
    # _symlink_if_needed(md5sums_file_name, network_imagenet_dir)
    md5sums_file = root / md5sums_file_name
    if not md5sums_file.exists():
        shutil.copyfile(network_imagenet_dir / md5sums_file_name, md5sums_file)
        md5sums_file.chmod(0o755)

    if split == "train":
        train_dir = root / "train"
        train_dir.mkdir(exist_ok=True, parents=True)
        train_archive = network_imagenet_dir / train_archive_file_name
        previously_extracted_dirs_file = train_dir / ".previously_extracted_dirs.txt"
        _extract_train_archive(
            train_archive=train_archive,
            train_dir=train_dir,
            previously_extracted_dirs_file=previously_extracted_dirs_file,
        )
        if previously_extracted_dirs_file.exists():
            previously_extracted_dirs_file.unlink()

        # OR: could just reuse the equivalent-ish from torchvision, but which doesn't support
        # resuming after an interrupt.
        # from torchvision.datasets.imagenet import parse_train_archive
        # parse_train_archive(root, file=train_archive_file_name, folder="train")
    else:
        from torchvision.datasets.imagenet import (
            load_meta_file,
            parse_devkit_archive,
            parse_val_archive,
        )

        parse_devkit_archive(root, file=devkit_file_name)
        wnids = load_meta_file(root)[1]
        val_dir = root / "val"
        if not val_dir.exists():
            logger.debug(f"Extracting ImageNet test set to {val_dir}")
            parse_val_archive(root, file=val_archive_file_name, wnids=wnids)
            return

        logger.debug(f"listing the contents of {val_dir}")
        children = list(val_dir.iterdir())

        if not children:
            logger.debug(f"Extracting ImageNet test set to {val_dir}")
            parse_val_archive(root, file=val_archive_file_name, wnids=wnids)
            return

        if all(child.is_dir() for child in children):
            logger.info("Validation split already extracted. Skipping.")
            return

        logger.warning(
            f"Incomplete extraction of the ImageNet test set in {val_dir}, deleting it and extracting again."
        )
        shutil.rmtree(root / "val", ignore_errors=False)
        parse_val_archive(root, file=val_archive_file_name, wnids=wnids)

        # val_dir = root / "val"
        # val_dir.mkdir(exist_ok=True, parents=True)
        # with tarfile.open(network_imagenet_dir / val_archive_file_name) as val_tarfile:
        #     val_tarfile.extractall(val_dir)


def _extract_train_archive(
    *, train_archive: Path, train_dir: Path, previously_extracted_dirs_file: Path
) -> None:
    # The ImageNet train archive is a tarfile of tarfiles (one for each class).
    logger.debug("Extracting the ImageNet train archive using Olexa's tar magic in python form...")
    train_dir.mkdir(exist_ok=True, parents=True)

    # Save a small text file or something that tells us which subdirs are
    # done extracting so we can just skip ahead to the right directory?
    previously_extracted_dirs: set[str] = set()

    if previously_extracted_dirs_file.exists():
        previously_extracted_dirs = set(
            stripped_line
            for line in previously_extracted_dirs_file.read_text().splitlines()
            if (stripped_line := line.strip())
        )
        if len(previously_extracted_dirs) == 1000:
            logger.info("Train archive already fully extracted. Skipping.")
            return
        logger.debug(
            f"{len(previously_extracted_dirs)} directories have already been fully extracted."
        )
        previously_extracted_dirs_file.write_text(
            "\n".join(sorted(previously_extracted_dirs)) + "\n"
        )

    elif len(list(train_dir.iterdir())) == 1000:
        logger.info("Train archive already fully extracted. Skipping.")
        return

    with tarfile.open(train_archive, mode="r") as train_tarfile:
        for member in tqdm.tqdm(
            train_tarfile,
            total=1000,  # hard-coded here, since we know there are 1000 folders.
            desc="Extracting train archive",
            unit="Directories",
            position=0,
        ):
            if member.name in previously_extracted_dirs:
                continue

            buffer = train_tarfile.extractfile(member)
            assert buffer is not None

            class_subdir = train_dir / member.name.replace(".tar", "")
            class_subdir_existed = class_subdir.exists()
            if class_subdir_existed:
                # Remove all the (potentially partially constructed) files in the directory.
                logger.debug(f"Removing partially-constructed dir {class_subdir}")
                shutil.rmtree(class_subdir, ignore_errors=False)
            else:
                class_subdir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(fileobj=buffer, mode="r|*") as class_tarfile:
                class_tarfile.extractall(class_subdir, filter="data")

                # Alternative: .extractall with a list of members to extract:
                # members = sub_tarfile.getmembers()  # note: loads the full archive.
                # if not files_in_subdir:
                #     members_to_extract = members
                # else:
                #     members_to_extract = [m for m in members if m.name not in files_in_subdir]
                # if members_to_extract:
                #     sub_tarfile.extractall(subdir, members=members_to_extract, filter="data")

            assert member.name not in previously_extracted_dirs
            previously_extracted_dirs.add(member.name)
            with previously_extracted_dirs_file.open("a") as f:
                f.write(f"{member.name}\n")


def main():
    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", handlers=[rich.logging.RichHandler()]
    )
    datamodule = ImageNetDataModule()
    start = time.time()
    datamodule.prepare_data()
    datamodule.setup("fit")
    dl = datamodule.train_dataloader()
    _batch = next(iter(dl))
    end = time.time()
    print(f"Prepared imagenet in {end-start:.2f}s.")


if __name__ == "__main__":
    main()

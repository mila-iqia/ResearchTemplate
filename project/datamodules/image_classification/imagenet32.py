from __future__ import annotations

import copy
import pickle
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from logging import getLogger
from pathlib import Path
from typing import ClassVar, Literal

import gdown
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2 as transforms

from project.datamodules.vision import VisionDataModule
from project.utils.env_vars import DATA_DIR, SCRATCH
from project.utils.typing_utils import C, H, W

logger = getLogger(__name__)


def imagenet32_normalization():
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class ImageNet32Dataset(VisionDataset):
    """Downsampled ImageNet 32x32 Dataset."""

    url: ClassVar[str] = "https://drive.google.com/uc?id=1XAlD_wshHhGNzaqy8ML-Jk0ZhAm8J5J_"
    md5: ClassVar[str] = "64cae578416aebe1576729ee93e41c25"
    archive_filename: ClassVar[str] = "imagenet32.tar.gz"

    dims: ClassVar[tuple[C, H, W]] = (C(3), H(32), W(32))

    def __init__(
        self,
        root: str | Path,
        readonly_datasets_dir: str | Path | None = None,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ):
        super().__init__(str(root), transform=transform, target_transform=target_transform)
        self.base_folder = "imagenet32"
        self.train = train  # training set or test set
        self.split = "train" if self.train else "val"
        self.split_folder = f"out_data_{self.split}"
        # TODO: Look for the archive in this directory before downloading it.
        self.readonly_datasets_dir = (
            Path(readonly_datasets_dir).expanduser().absolute() if readonly_datasets_dir else None
        )

        self._data_loaded = False
        self.data: np.ndarray
        self.targets: np.ndarray

        if download:
            self._download_dataset()
            self._load_dataset()
        else:
            try:
                self._load_dataset()
            except FileNotFoundError as err:
                raise RuntimeError(
                    f"Missing the files for ImageNet32 {self.split} dataset, run this with "
                    f"`download=True` first."
                ) from err

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _download_dataset(self) -> None:
        archive_path = (Path(self.root) / self.archive_filename).absolute()
        extracted_path = (Path(self.root) / self.base_folder).absolute()
        root_path = Path(self.root).absolute()

        def extract_archive_in_root():
            # Check if the archive is already extracted somehow?
            logger.info(f"Extracting archive {archive_path} to {root_path}")
            shutil.unpack_archive(archive_path, extract_dir=str(root_path))

        if extracted_path.exists():
            logger.info(f"Extraction path {extracted_path} already exists.")
            try:
                self._load_dataset()
                logger.info(f"Archive already downloaded and extracted to {extracted_path}")
            except Exception as exc:
                # Unable to load the dataset, for some reason. Re-extract it.
                logger.info(f"Unable to load the dataset from {extracted_path}: {exc}\n")
                logger.info("Re-extracting the archive, which will overwrite the files present.")
                extract_archive_in_root()
            return

        if archive_path.exists():
            extract_archive_in_root()
            return
        if (
            self.readonly_datasets_dir
            and (self.readonly_datasets_dir / self.archive_filename).exists()
        ):
            readonly_archive_path = self.readonly_datasets_dir / self.archive_filename
            logger.info(f"Found the archive at {readonly_archive_path}")
            logger.info(f"Copying archive from {readonly_archive_path} -> {archive_path}")
            shutil.copyfile(src=readonly_archive_path, dst=archive_path, follow_symlinks=False)
            extract_archive_in_root()
            return

        if not archive_path.exists():
            logger.info(f"Downloading the archive to {archive_path}")
            # TODO: This uses the ~/.cache/gdown/ directory, which is not great!
            gdown.cached_download(
                url=self.url,
                md5=self.md5,
                path=str(archive_path),
                quiet=False,
                postprocess=gdown.extractall,
            )

    def _load_dataset(self):
        if self._data_loaded:
            logger.info("Data already loaded. Skipping.")
            return
        data = []
        targets = []

        # Load the picked numpy arrays
        logger.info(f"Loading ImageNet32 {self.split} dataset...")
        for i in range(1, 11):
            file_name = "train_data_batch_" + str(i)
            file_path = Path(self.root, self.base_folder, self.split_folder, file_name).absolute()
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])
        self.targets = np.array(targets) - 1
        # self.targets = [t - 1 for t in self.targets]
        self.data = np.vstack(data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        logger.info(f"Loaded {len(self.data)} images from ImageNet32 {self.split} split")
        self._data_loaded = True


class ImageNet32DataModule(VisionDataModule):
    """TODO: Add a `val_split` argument, that supports a value of `0`."""

    name: ClassVar[str] = "imagenet32"
    dataset_cls: ClassVar[type[ImageNet32Dataset]] = ImageNet32Dataset
    dims: ClassVar[tuple[C, H, W]] = (C(3), H(32), W(32))
    num_classes: ClassVar[int] = 1000

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        readonly_datasets_dir: str | Path | None = SCRATCH,
        val_split: int | float = -1,
        num_images_per_val_class: int | None = 50,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
    ) -> None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
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
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            # extra kwargs
            readonly_datasets_dir=readonly_datasets_dir,
        )
        self.num_images_per_val_class = num_images_per_val_class
        if self.val_split == -1 and self.num_images_per_val_class is None:
            raise ValueError(
                "Can't have both `val_split` and `num_images_per_val_class` set to `None`!"
            )
        if val_split != -1 and self.num_images_per_val_class is not None:
            logger.warning(
                "Both `num_images_per_val_class` and `val_split` are set. "
                "Ignoring value of `num_images_per_val_class` and setting it to None."
            )
            self.num_images_per_val_class = None

        self.dataset_train: ImageNet32Dataset | Subset
        self.dataset_val: ImageNet32Dataset | Subset
        self.dataset_test: ImageNet32Dataset | Subset

    @property
    def num_samples(self) -> int:
        return len(self.dataset_train)

    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        super().prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        # """Creates train, val, and test dataset."""
        if stage:
            logger.debug(f"Setting up for stage {stage}")
        else:
            logger.debug("Setting up for all stages")

        if stage in ["fit", "validate", None]:
            base_dataset = self.dataset_cls(self.data_dir, **self.train_kwargs)
            assert len(base_dataset) == 1_281_159

            base_dataset_train = copy.deepcopy(base_dataset)
            base_dataset_train.transform = self.train_transforms
            base_dataset_train.data = base_dataset.data
            base_dataset_train.targets = base_dataset.targets

            base_dataset_valid = copy.deepcopy(base_dataset)
            base_dataset_valid.transform = self.val_transforms
            base_dataset_valid.data = base_dataset.data
            base_dataset_valid.targets = base_dataset.targets

            if self.num_images_per_val_class is not None:
                train_indices, val_indices = get_train_val_indices(
                    dataset_labels=base_dataset.targets,
                    nb_imgs_in_val=self.num_images_per_val_class,
                    split_seed=self.seed,
                )
                self.dataset_train = Subset(base_dataset_train, train_indices)
                self.dataset_val = Subset(base_dataset_valid, val_indices)
            else:
                self.dataset_train = self._split_dataset(base_dataset_train, train=True)
                self.dataset_val = self._split_dataset(base_dataset_valid, train=False)

        if stage in ["test", "predict", None]:
            test_transforms = self.test_transforms or self.default_transforms()
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
            + ([imagenet32_normalization()] if self.normalize else [])
        )

    def train_dataloader(self) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def _split_dataset(self, dataset: ImageNet32Dataset, train: bool = True) -> Subset:
        assert self.val_split >= 0
        split_dataset = super()._split_dataset(dataset, train=train)
        assert isinstance(split_dataset, Subset)
        return split_dataset


# TODO: Do something like this to partition the train and val sets, instead of using a val_fraction


def get_train_val_indices(
    dataset_labels: Sequence[int] | np.ndarray,
    nb_imgs_in_val: int,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Keeps the first `nb_imgs_in_val` images of each class in the validation set."""
    val_indices: list[int] = []
    train_indices: list[int] = []

    index_and_label = np.array(list(enumerate(dataset_labels)))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(index_and_label)

    n_val_samples_per_class = defaultdict(int)
    for index, y in index_and_label:
        if n_val_samples_per_class[y] < nb_imgs_in_val:
            val_indices.append(index)
            n_val_samples_per_class[y] += 1
        else:
            train_indices.append(index)
    return train_indices, val_indices


def imagenet32_train_transforms():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=32, padding=4, padding_mode="edge"),
            imagenet32_normalization(),
        ]
    )

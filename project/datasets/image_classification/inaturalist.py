from __future__ import annotations

import warnings
from collections.abc import Callable
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, ClassVar, Literal

import torchvision.transforms as T
from torchvision.datasets import INaturalist, VisionDataset

from project.datamodules.vision import VisionDataModule
from project.utils.env_vars import DATA_DIR, NUM_WORKERS, SLURM_TMPDIR
from project.utils.typing_utils import C, H, W

logger = get_logger(__name__)


Version2021 = Literal["2021_train", "2021_train_mini", "2021_valid"]
Version2017_2019 = Literal["2017", "2018", "2019"]
Target2017_2019 = Literal["full", "super"]
Target2021 = Literal["full", "kingdom", "phylum", "class", "order", "family", "genus"]

TargetType = Target2017_2019 | Target2021
Version = Version2017_2019 | Version2021


def inat_dataset_dir() -> Path:
    network_dir = Path("/network/datasets/inat")
    if not network_dir.exists():
        raise NotImplementedError("For now this assumes that we're running on the Mila cluster.")
    return network_dir


class INaturalistDataModule(VisionDataModule):
    name: str | None = "inaturalist"
    """Dataset name."""

    dataset_cls: ClassVar[type[VisionDataset]] = INaturalist
    """Dataset class to use."""

    dims: tuple[C, H, W] = (C(3), H(224), W(224))
    """A tuple describing the shape of the data."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        val_split: int | float = 0.1,
        num_workers: int = NUM_WORKERS,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Callable[..., Any] | None = None,
        val_transforms: Callable[..., Any] | None = None,
        test_transforms: Callable[..., Any] | None = None,
        version: Version = "2021_train",
        target_type: TargetType | list[TargetType] = "full",
        **kwargs,
    ) -> None:
        # assuming that we're on the Mila cluster atm.
        self.network_dir = inat_dataset_dir()
        assert SLURM_TMPDIR, "assuming that we're on a compute node."
        slurm_tmpdir = SLURM_TMPDIR
        default_data_dir = slurm_tmpdir / "data"
        if data_dir is None:
            data_dir = default_data_dir
        else:
            data_dir = Path(data_dir)
            if not data_dir.is_relative_to(slurm_tmpdir):
                warnings.warn(
                    RuntimeWarning(
                        f"Ignoring the chosen data dir {data_dir}, as it is not under "
                        f"$SLURM_TMPDIR! Using {default_data_dir} instead."
                    )
                )
                data_dir = default_data_dir
        data_dir.mkdir(exist_ok=True, parents=True)
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
            # Extra args:
            version=version,
            target_type=target_type,
            **kwargs,
        )
        self.version = version
        self.target_type = target_type

        # NOTE: Setting this attribute will make this compatible with the
        # ImageClassificationDataModule protocol.
        self.num_classes: int | None

        if not isinstance(target_type, list):
            self.num_classes = None
        # todo: double-check that the 2021_train split also has 10_000 classes.
        if version in ["2021_train_mini", "2021_train"] and target_type == "full":
            self.num_classes = 10_000
        if isinstance(train_transforms, T.Compose):
            channels = 3
            for t in train_transforms.transforms:
                if isinstance(t, T.RandomResizedCrop):
                    self.dims = (C(channels), H(t.size[0]), W(t.size[1]))
                if isinstance(t, T.CenterCrop):
                    self.dims = (C(channels), H(t.size[0]), W(t.size[1]))
                if isinstance(t, T.Resize):
                    h = t.size if isinstance(t.size, int) else t.size[0]
                    w = t.size if isinstance(t.size, int) else t.size[1]
                    self.dims = (C(channels), H(h), W(w))

        self.train_kwargs["version"] = self.version
        self.test_kwargs["version"] = "2021_valid"

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # Make symlinks in SLURM_TMPDIR pointing to the archives on the cluster.
        # Note: We have the same archives, but with a slightly different name.
        # This should save time, compared to copying the archives to $SLURM_TMPDIR and then
        # extracting them.
        for archive_name_in_torchvision, archive_name_on_network in {
            "2021_train.tgz": "train.tar.gz",
            "2021_train_mini.tgz": "train_mini.tar.gz",
            "2021_valid.tgz": "val.tar.gz",
        }.items():
            symlink_in_tmpdir = Path(self.data_dir) / archive_name_in_torchvision
            file_on_network = self.network_dir / archive_name_on_network
            if not symlink_in_tmpdir.exists():
                symlink_in_tmpdir.symlink_to(file_on_network)

        try:
            logger.debug(f"Checking if the dataset has already been created in {self.data_dir}.")
            self.dataset_cls(str(self.data_dir), download=False, **self.EXTRA_ARGS)
        except RuntimeError:
            logger.debug(f"The dataset has not already been created in {self.data_dir}.")
            pass
        else:
            logger.debug(f"The dataset has already been downloaded in {self.data_dir}.")
            return

        return super().prepare_data(*args, **kwargs)

    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""
        return T.Compose(
            [
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        )

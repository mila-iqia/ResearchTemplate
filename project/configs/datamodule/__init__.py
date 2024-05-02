import os
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from pathlib import Path

import torch
from hydra_zen import hydrated_dataclass, instantiate
from torch import Tensor

from project.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImageNet32DataModule,
    MNISTDataModule,
    RlDataModule,
    VisionDataModule,
)
from project.datamodules.vision.cifar10 import cifar10_train_transforms
from project.datamodules.vision.imagenet32 import imagenet32_train_transforms
from project.datamodules.vision.inaturalist import INaturalistDataModule, TargetType, Version
from project.datamodules.vision.mnist import mnist_train_transforms
from project.datamodules.vision.moving_mnist import MovingMnistDataModule

FILE = Path(__file__)
REPO_ROOTDIR = FILE.parent
for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
SLURM_JOB_ID: int | None = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)

logger = get_logger(__name__)


TORCHVISION_DIR: Path | None = None

_torchvision_dir = Path("/network/datasets/torchvision")
if _torchvision_dir.exists() and _torchvision_dir.is_dir():
    TORCHVISION_DIR = _torchvision_dir


if not SLURM_TMPDIR and SLURM_JOB_ID is not None:
    # This can happens when running the integrated VSCode terminal with `mila code`!
    _slurm_tmpdir = Path(f"/Tmp/slurm.{SLURM_JOB_ID}.0")
    if _slurm_tmpdir.exists():
        SLURM_TMPDIR = _slurm_tmpdir
SCRATCH = Path(os.environ["SCRATCH"]) if "SCRATCH" in os.environ else None
DATA_DIR = Path(os.environ.get("DATA_DIR", (SLURM_TMPDIR or SCRATCH or REPO_ROOTDIR) / "data"))

NUM_WORKERS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get(
            "SLURM_CPUS_ON_NODE",
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else torch.multiprocessing.cpu_count(),
        ),
    )
)
logger = get_logger(__name__)


Transform = Callable[[Tensor], Tensor]


@dataclass
class DataModuleConfig: ...


@hydrated_dataclass(target=VisionDataModule, populate_full_signature=True)
class VisionDataModuleConfig(DataModuleConfig):
    data_dir: str | None = str(TORCHVISION_DIR or DATA_DIR)
    val_split: int | float = 0.1  # NOTE: reduced from default of 0.2
    num_workers: int = NUM_WORKERS
    normalize: bool = True  # NOTE: Set to True by default instead of False
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = True  # NOTE: Set to True by default instead of False.
    pin_memory: bool = True  # NOTE: Set to True by default instead of False.
    drop_last: bool = False

    __call__ = instantiate


# todo: look into this to avoid having to make dataclasses with no fields just to call a function..
from hydra_zen import store, zen  # noqa


# FIXME: This is dumb!
@hydrated_dataclass(target=mnist_train_transforms)
class MNISTTrainTransforms: ...


@hydrated_dataclass(target=MNISTDataModule, populate_full_signature=True)
class MNISTDataModuleConfig(VisionDataModuleConfig):
    normalize: bool = True
    batch_size: int = 128
    train_transforms: MNISTTrainTransforms = field(default_factory=MNISTTrainTransforms)


@hydrated_dataclass(target=FashionMNISTDataModule, populate_full_signature=True)
class FashionMNISTDataModuleConfig(MNISTDataModuleConfig): ...


@hydrated_dataclass(target=cifar10_train_transforms)
class Cifar10TrainTransforms: ...


@hydrated_dataclass(target=CIFAR10DataModule, populate_full_signature=True)
class CIFAR10DataModuleConfig(VisionDataModuleConfig):
    train_transforms: Cifar10TrainTransforms = field(default_factory=Cifar10TrainTransforms)
    # Overwriting this one:
    batch_size: int = 128


@hydrated_dataclass(target=imagenet32_train_transforms)
class ImageNet32TrainTransforms: ...


@hydrated_dataclass(target=ImageNet32DataModule, populate_full_signature=True)
class ImageNet32DataModuleConfig(VisionDataModuleConfig):
    data_dir: Path = ((SCRATCH / "data") if SCRATCH else DATA_DIR) / "imagenet32"

    val_split: int | float = -1
    num_images_per_val_class: int = 50  # Slightly different.
    normalize: bool = True
    train_transforms: ImageNet32TrainTransforms = field(default_factory=ImageNet32TrainTransforms)


@hydrated_dataclass(target=RlDataModule, populate_full_signature=False)
class RlDataModuleConfig(DataModuleConfig):
    env: str = "CartPole-v1"
    episodes_per_epoch: int = 100
    batch_size: int = 1


@hydrated_dataclass(target=INaturalistDataModule, populate_full_signature=True)
class INaturalistDataModuleConfig(VisionDataModuleConfig):
    data_dir: Path | None = None
    version: Version = "2021_train"
    target_type: TargetType | list[TargetType] = "full"


@hydrated_dataclass(target=MovingMnistDataModule, populate_full_signature=True)
class MovingMnistDataModuleConfig(VisionDataModuleConfig):
    data_dir: Path | None = SLURM_TMPDIR

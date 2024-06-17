from __future__ import annotations

import contextlib
import os
import shutil
import tarfile
import time
from collections.abc import Callable
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar, Concatenate, Literal, TypeVar

import torch
import tqdm
from torchvision.datasets import ImageNet

from project.configs.datamodule import DATA_DIR
from project.datamodules.vision.base import VisionDataModule
from project.utils.types import C, H, StageStr, W
from project.utils.types.protocols import Module

logger = get_logger(__name__)
ImageNetType = TypeVar("ImageNetType", bound=ImageNet)


@contextlib.contextmanager
def change_directory(path: Path):
    curdir = Path.cwd()
    os.chdir(path)
    yield
    os.chdir(curdir)


class ImageNetDataModule(VisionDataModule):
    name: ClassVar[str] = "imagenet"
    """Dataset name."""

    dataset_cls: ClassVar[type[ImageNet]] = ImageNet
    """Dataset class to use."""

    dims: tuple[C, H, W] = (C(3), H(224), W(224))
    """A tuple describing the shape of the data."""

    num_classes: ClassVar[int] = 1000

    def __init__(
        self,
        root: str | Path = DATA_DIR,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.root = Path(root)

    def prepare_data(self) -> None:
        network_imagenet_dir = Path("/network/datasets/imagenet")
        assert network_imagenet_dir.exists()
        prepare_imagenet(
            self.root, network_imagenet_dir=network_imagenet_dir, split="train", **self.EXTRA_ARGS
        )
        prepare_imagenet(
            self.root, network_imagenet_dir=network_imagenet_dir, split="val", **self.EXTRA_ARGS
        )

    def setup(self, stage: StageStr | None = None) -> None:
        super().setup(stage)

    def default_transforms(self) -> Module[[torch.Tensor], torch.Tensor]:
        from torchvision.models.resnet import ResNet152_Weights

        return ResNet152_Weights.IMAGENET1K_V1.transforms


def prepare_imagenet[**P](
    root: str | Path,
    split: Literal["train", "val"] = "train",
    network_imagenet_dir: Path = Path("/network/datasets/imagenet"),
    _dataset: Callable[Concatenate[str, Literal["train", "val"], P], ImageNet] = ImageNet,
    *args: P.args,
    **kwargs: P.kwargs,
) -> ImageNet:
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
    root = Path(root)
    if not network_imagenet_dir.exists():
        raise NotImplementedError(
            f"Assuming that we're running on the Mila cluster where {network_imagenet_dir} exists for now."
        )
    val_archive_file_name = "ILSVRC2012_img_val.tar"
    train_archive_file_name = "ILSVRC2012_img_train.tar"
    devkit_file_name = "ILSVRC2012_devkit_t12.tar.gz"
    md5sums_file_name = "md5sums"

    def _symlink_if_needed(filename: str, network_imagenet_dir: Path):
        symlink = root / filename
        if not symlink.exists():
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

    logger.info("Extracting the ImageNet archives using Olexa's tar magic in python form...")

    if split == "train":
        train_dir = root / "train"
        train_dir.mkdir(exist_ok=True, parents=True)

        # The ImageNet train archive is a tarfile of tarfiles (one for each class).
        with tarfile.open(network_imagenet_dir / train_archive_file_name) as train_tarfile:
            for member in tqdm.tqdm(
                train_tarfile,
                total=1000,  # hard-coded here, since we know there are 1000 folders.
                desc="Extracting train archive",
                unit="Directories",
                position=0,
            ):
                buffer = train_tarfile.extractfile(member)
                assert buffer is not None
                subdir = train_dir / member.name.replace(".tar", "")
                subdir.mkdir(mode=0o755, parents=True, exist_ok=True)
                files_in_subdir = set(p.name for p in subdir.iterdir())
                with tarfile.open(fileobj=buffer, mode="r|*") as sub_tarfile:
                    for tarinfo in sub_tarfile:
                        if tarinfo.name in files_in_subdir:
                            # Image file is already in the directory.
                            continue
                        sub_tarfile.extract(tarinfo, subdir)

    else:
        val_dir = root / "val"
        val_dir.mkdir(exist_ok=True, parents=True)
        with tarfile.open(network_imagenet_dir / val_archive_file_name) as val_tarfile:
            val_tarfile.extractall(val_dir)

    return _dataset(str(root), split, *args, **kwargs)


def main():
    slurm_tmpdir = Path(os.environ["SLURM_TMPDIR"])
    datamodule = ImageNetDataModule(slurm_tmpdir)
    start = time.time()
    datamodule.prepare_data()
    datamodule.setup("fit")
    dl = datamodule.train_dataloader()
    _batch = next(iter(dl))
    end = time.time()
    print(f"Prepared imagenet in {end-start:.2f}s.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MovingMNIST, VisionDataset
from typing_extensions import ParamSpec

from .bases.vision import C, H, VisionDataModule, W

P = ParamSpec("P")


class Squeeze(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        # [B, 10, 1, 64, 64] -> [B, 10, 64, 64]
        return input.squeeze()


class SplitVideoIntoBeforeAfter(torch.nn.Module):
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        if input.ndim == 3:
            length = input.shape[0]
            # [10, 64, 64] -> ([5, 64, 64], [5, 64, 64])
            return input[: length // 2], input[length // 2 :]
        assert input.ndim == 4
        if input.shape[0] == 1:  # There is still that useless "channels" dimension:
            length = input.shape[1]
            # [1, 10, 64, 64] -> ([1, 5, 64, 64], [1, 5, 64, 64])
            return input[:, : length // 2], input[:, length // 2 :]
        else:
            # Transform is being applied on a batch (?)
            # [B, 10, 64, 64] -> ([B, 5, 64, 64], [B, 5, 64, 64])
            length = input.shape[1]
            return input[:, : length // 2], input[:, length // 2 :]

        # length = input.shape[0]
        # # [10, 64, 64] -> ([5, 64, 64], [5, 64, 64])
        # return input[: length // 2], input[length // 2 :]


def default_transforms():
    return transforms.Compose([Squeeze(), SplitVideoIntoBeforeAfter()])


class MovingMnistDataModule(VisionDataModule[tuple[Tensor, Tensor]]):
    name: ClassVar[str] = "moving_mnist"
    """Dataset name."""

    dataset_cls: ClassVar[type[VisionDataset]] = MovingMNIST
    """Dataset class to use."""

    dims: ClassVar[tuple[C, H, W]] = (C(5), H(64), W(64))
    """A tuple describing the shape of the data."""

    y_dims: ClassVar[tuple[C, H, W]] = (C(5), H(64), W(64))

    def default_transforms(self) -> Callable[..., Any]:
        # Removes the dimension of size 1, otherwise the values have shape [B, 10, 1, 64, 64]
        return default_transforms()

    # def default_transforms(self) -> Callable[..., Any]:
    #     return torchvision.transforms.ToTensor()

    def prepare_data(self) -> None:
        network_dataset_dir = Path("/network/datasets/movingmnist.var/movingmnist_torchvision")
        if network_dataset_dir.exists():
            from project.configs.datamodule import SLURM_TMPDIR

            assert SLURM_TMPDIR is not None
            data_dir = Path(self.data_dir)
            assert data_dir.is_relative_to(SLURM_TMPDIR)
            new_dataset_dir = data_dir / "MovingMNIST"
            new_dataset_dir.mkdir(parents=False, exist_ok=True)
            for file in network_dataset_dir.iterdir():
                if file.name.startswith(".") or file.name == "scripts":
                    continue
                symlink = new_dataset_dir / file.name
                if not symlink.exists():
                    symlink.symlink_to(file)
        super().prepare_data()

from __future__ import annotations

from dataclasses import field
from logging import getLogger as get_logger
from typing import Iterable, Sequence, TypeVar, Union

import rich
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torchvision import transforms
from project.datamodules.image_classification import ImageClassificationDataModule


from .types import DM

logger = get_logger(__name__)

K = TypeVar("K")
T = TypeVar("T")
V = TypeVar("V", bound=Union[int, float])


def list_field(*values: T) -> list[T]:
    return field(default_factory=list(values).copy)


def is_trainable(layer: nn.Module) -> bool:
    return any(p.requires_grad for p in layer.parameters())


def named_trainable_parameters(module: nn.Module) -> Iterable[tuple[str, Parameter]]:
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def get_device(mod: nn.Module) -> torch.device:
    return next(p.device for p in mod.parameters() if p.requires_grad)


def _remove_normalization_from_transforms(datamodule: ImageClassificationDataModule) -> None:
    transform_properties = (
        datamodule.train_transforms,
        datamodule.val_transforms,
        datamodule.test_transforms,
    )
    for transform_list in transform_properties:
        if transform_list is None:
            continue
        assert isinstance(transform_list, transforms.Compose)
        if isinstance(transform_list.transforms[-1], transforms.Normalize):
            t = transform_list.transforms.pop(-1)
            logger.info(f"Removed normalization transform {t} since datamodule.normalize=False")
        if any(isinstance(t, transforms.Normalize) for t in transform_list.transforms):
            raise RuntimeError(
                f"Unable to remove all the normalization transforms from datamodule {datamodule}: "
                f"{transform_list}"
            )


def validate_datamodule(datamodule: DM) -> DM:
    """Checks that the transforms / things are setup correctly.

    Returns the same datamodule.
    """
    if isinstance(datamodule, ImageClassificationDataModule) and not datamodule.normalize:
        _remove_normalization_from_transforms(datamodule)
    else:
        # todo: maybe check that the normalization transform is present everywhere?
        pass
    return datamodule


def tile_batch(v: Tensor, n: int) -> Tensor:
    return v.tile([n, *(1 for _ in v.shape[1:])])


def repeat_batch(v: Tensor, n: int) -> Tensor:
    """Repeats the elements of tensor `v` `n` times along the batch dimension:

    Example:

    input:  [[1, 2, 3], [4, 5, 6]] of shape=(2, 3), n = 2
    output: [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]] of shape=(4, 3)

    >>> import torch
    >>> input = torch.as_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> repeat_batch(input, 2).tolist()
    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]
    """
    b = v.shape[0]
    batched_v = v.unsqueeze(1).expand([b, n, *v.shape[1:]])  # [B, N, ...]
    flattened_batched_v = batched_v.reshape([b * n, *v.shape[1:]])  # [N*B, ...]
    return flattened_batched_v


def split_batch(batched_v: Tensor, n: int) -> Tensor:
    """Reshapes the output of `repeat_batch` from shape [B*N, ...] back to a shape of [B, N, ...]

    Example:

    input: [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]
        shape=(4, 3)
    output: [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]],
        shape=(2, 2, 3)

    >>> import numpy as np
    >>> input = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [4.0, 5.0, 6.0], [4.1, 5.1, 6.1]])
    >>> split_batch(input, 2).tolist()
    [[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]]
    """
    assert batched_v.shape[0] % n == 0
    # [N*B, ...] -> [N, B, ...]
    return batched_v.reshape([-1, n, *batched_v.shape[1:]])


# from lightning.utilities.rank_zero import rank_zero_only


# @rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "algorithm",
        "network",
        "datamodule",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    TAKEN FROM https://github.com/ashleve/lightning-hydra-template/blob/6a92395ed6afd573fa44dd3a054a603acbdcac06/src/utils/__init__.py#L56

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are
        printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for f in print_order:
        queue.append(f) if f in config else logger.info(f"Field '{f}' not found in config")

    for f in config:
        if f not in queue:
            queue.append(f)

    for f in queue:
        branch = tree.add(f, style=style, guide_style=style)

        config_group = config[f]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    # with open("config_tree.log", "w") as file:
    #     rich.print(tree, file=file)

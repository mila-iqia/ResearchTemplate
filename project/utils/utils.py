from __future__ import annotations

import functools
import typing
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import field
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal

import rich
import rich.syntax
import rich.tree
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import transforms

from project.utils.types.protocols import (
    DataModule,
    Module,
)

from .types import NestedDict, NestedMapping

logger = get_logger(__name__)


# todo: doesn't work? keeps logging each time!
@functools.cache
def log_once(message: str, level: int) -> None:
    """Logs a message once per logger instance. The message is logged at the specified level.

    Args:
        message: The message to log.
        level: The logging level to use.
    """
    logger.log(level=level, msg=message, stacklevel=2)


def get_shape_ish(t: Tensor) -> tuple[int | Literal["?"], ...]:
    if not t.is_nested:
        return t.shape
    dim_sizes = []
    for dim in range(t.ndim):
        try:
            dim_sizes.append(t.size(dim))
        except RuntimeError:
            dim_sizes.append("?")
    return tuple(dim_sizes)


def relative_if_possible(p: Path) -> Path:
    try:
        return p.relative_to(Path.cwd())
    except ValueError:
        return p.absolute()


def get_log_dir(trainer: Trainer | None) -> Path:
    """Gives back the default directory to use when `trainer.log_dir` is None (no logger used?)"""
    # TODO: This isn't great.. It could probably be a property on the Algorithm class or
    # customizable somehow.
    # ALSO: This
    if trainer:
        if trainer.logger and trainer.logger.log_dir:
            return Path(trainer.logger.log_dir)
        if trainer.log_dir:
            return Path(trainer.log_dir)
    base = Path(trainer.default_root_dir) if trainer else Path.cwd() / "logs"
    log_dir = base / "default"
    warnings.warn(
        RuntimeWarning(
            f"Using the default log directory of {log_dir} because the trainer.log_dir is None. "
            f"Consider setting `trainer.logger` (e.g. `trainer.logger=wandb`) so this is set!"
        )
    )
    return log_dir


def list_field[T](*values: T) -> list[T]:
    return field(default_factory=list(values).copy)


def is_trainable(layer: Module) -> bool:
    return any(p.requires_grad for p in layer.parameters())


def named_trainable_parameters(module: Module) -> Iterable[tuple[str, Parameter]]:
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def get_device(mod: Module) -> torch.device:
    return next(p.device for p in mod.parameters())


def get_devices(mod: Module) -> set[torch.device]:
    return set(p.device for p in mod.parameters())


if typing.TYPE_CHECKING:
    from project.datamodules.image_classification.image_classification import (
        ImageClassificationDataModule,
    )


# todo: shouldn't be here, should be done in `VisionDataModule` or in the configs:
# If `normalize=False`, and there is a normalization transform in the train transforms, then an
# error should be raised.
def _remove_normalization_from_transforms(
    datamodule: ImageClassificationDataModule,
) -> None:
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


def validate_datamodule[DM: DataModule | LightningDataModule](datamodule: DM) -> DM:
    """Checks that the transforms / things are setup correctly.

    Returns the same datamodule.
    """
    from project.datamodules.image_classification.image_classification import (
        ImageClassificationDataModule,
    )

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
        config: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
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


def flatten[K, V](nested: NestedMapping[K, V]) -> dict[tuple[K, ...], V]:
    """Flatten a dictionary of dictionaries. The returned dictionary's keys are tuples, one entry
    per layer.

    >>> flatten({"a": {"b": 2, "c": 3}, "c": {"d": 3, "e": 4}})
    {('a', 'b'): 2, ('a', 'c'): 3, ('c', 'd'): 3, ('c', 'e'): 4}
    """
    flattened: dict[tuple[K, ...], V] = {}
    for k, v in nested.items():
        if isinstance(v, Mapping):
            for subkeys, subv in flatten(v).items():
                collision_key = (k, *subkeys)
                assert collision_key not in flattened
                flattened[collision_key] = subv
        else:
            flattened[(k,)] = v
    return flattened


def unflatten[K, V](flattened: dict[tuple[K, ...], V]) -> NestedDict[K, V]:
    """Unflatten a dictionary back into a possibly nested dictionary.

    >>> unflatten({('a', 'b'): 2, ('a', 'c'): 3, ('c', 'd'): 3, ('c', 'e'): 4})
    {'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}}
    """
    nested: NestedDict[K, V] = {}
    for keys, value in flattened.items():
        sub_dictionary = nested
        for part in keys[:-1]:
            assert isinstance(sub_dictionary, dict)
            sub_dictionary = sub_dictionary.setdefault(part, {})
        assert isinstance(sub_dictionary, dict)
        sub_dictionary[keys[-1]] = value
    return nested


def flatten_dict[V](nested: NestedMapping[str, V], sep: str = ".") -> dict[str, V]:
    """Flatten a dictionary of dictionaries. Joins different nesting levels with `sep` as
    separator.

    >>> flatten_dict({'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}})
    {'a.b': 2, 'a.c': 3, 'c.d': 3, 'c.e': 4}
    >>> flatten_dict({'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}}, sep="/")
    {'a/b': 2, 'a/c': 3, 'c/d': 3, 'c/e': 4}
    """
    return {sep.join(keys): value for keys, value in flatten(nested).items()}


def unflatten_dict[V](
    flattened: dict[str, V], sep: str = ".", recursive: bool = False
) -> NestedDict[str, V]:
    """Unflatten a dict into a possibly nested dict. Keys are split using `sep`.

    >>> unflatten_dict({'a.b': 2, 'a.c': 3, 'c.d': 3, 'c.e': 4})
    {'a': {'b': 2, 'c': 3}, 'c': {'d': 3, 'e': 4}}

    >>> unflatten_dict({'a': 2, 'b.c': 3})
    {'a': 2, 'b': {'c': 3}}

    NOTE: This function expects the input to be flat. It does *not* unflatten nested dicts:
    >>> unflatten_dict({"a": {"b.c": 2}})
    {'a': {'b.c': 2}}
    """
    return unflatten({tuple(key.split(sep)): value for key, value in flattened.items()})

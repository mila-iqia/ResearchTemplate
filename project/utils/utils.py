from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import TypeVar

import rich
import rich.syntax
import rich.tree
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf

from project.utils.typing_utils.protocols import (
    DataModule,
)

logger = get_logger(__name__)


def get_log_dir(trainer: Trainer | None) -> Path:
    """Gives back the default directory to use when `trainer.log_dir` is None (no logger used)."""
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
    logger.info(
        f"Using the default log directory of {log_dir} because a logger isn't being used."
        # f"Consider using a logger (e.g. with 'trainer.logger=wandb' on the command-line)."
    )
    return log_dir


DM = TypeVar("DM", bound=DataModule | LightningDataModule)


# todo: shouldn't be here, should be done in `VisionDataModule` or in the configs:
# If `normalize=False`, and there is a normalization transform in the train transforms, then an
# error should be raised.
# from lightning.utilities.rank_zero import rank_zero_only


# @rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "algorithm",
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
        if f in config:
            queue.append(f)
        else:
            logger.info(f"Field '{f}' not found in config")

    for f in config:
        if f not in queue:
            queue.append(f)

    for f in queue:
        if f not in config:
            logger.info(f"Field '{f}' not found in config")
            continue
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


def default_device() -> torch.device:
    """Returns the default device (GPU if available, else CPU)."""
    return torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

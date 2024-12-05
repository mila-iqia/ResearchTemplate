import random
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any, Literal, Optional

from omegaconf import OmegaConf

from project.utils.env_vars import get_constant

OmegaConf.register_new_resolver("constant", get_constant)

logger = get_logger(__name__)
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@dataclass
class Config:
    """The options required for a run. This dataclass acts as a structure for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    algorithm: Any
    """Configuration for the algorithm (a
    [LightningModule][lightning.pytorch.core.module.LightningModule]).

    It is suggested for this class to accept a `datamodule` and `network` as arguments. The
    instantiated datamodule and network will be passed to the algorithm's constructor.

    For more info, see the [instantiate_algorithm][project.main.instantiate_algorithm] function.
    """

    datamodule: Any | None = None
    """Configuration for the datamodule (dataset + transforms + dataloader creation).

    This should normally create a [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule].
    See the [MNISTDataModule][project.datamodules.image_classification.mnist.MNISTDataModule] for an example.
    """

    datamodule: Optional[Any] = None  # noqa
    """Configuration for the datamodule (dataset + transforms + dataloader creation).

    This should normally create a [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule].
    See the [MNISTDataModule][project.datamodules.image_classification.mnist.MNISTDataModule] for an example.
    """

    trainer: dict = field(default_factory=dict)
    """Keyword arguments for the Trainer constructor."""

    log_level: str = "info"
    """Logging level."""

    # Random seed.
    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    """Random seed for reproducibility.

    If None, a random seed is generated.
    """

    # Name for the experiment.
    name: str = "default"

    debug: bool = False

    verbose: bool = False

    ckpt_path: str | None = None
    """Path to a checkpoint to load the training state and resume the training run.

    This is the same as the `ckpt_path` argument in the `lightning.Trainer.fit` method.
    """

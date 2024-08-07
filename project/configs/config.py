import random
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any, Literal

from omegaconf import OmegaConf

from project.utils.env_vars import get_constant

OmegaConf.register_new_resolver("constant", get_constant)

logger = get_logger(__name__)
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@dataclass
class Config:
    """All the options required for a run. This dataclass acts as a schema for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    datamodule: Any
    """Configuration for the datamodule (dataset + transforms + dataloader creation)."""

    algorithm: Any
    """The hyper-parameters of the algorithm to use."""

    network: Any
    """The network to use."""

    trainer: dict = field(default_factory=dict)
    """Keyword arguments for the Trainer constructor."""

    log_level: str = "info"
    """Logging level."""

    # Random seed.
    seed: int | None = field(default_factory=lambda: random.randint(0, int(1e5)))
    """Random seed for reproducibility.

    If None, a random seed is generated.
    """

    # Name for the experiment.
    name: str = "default"

    debug: bool = False

    verbose: bool = False

import random
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Literal, Optional, Any
from hydra.core.config_store import ConfigStore

from project.algorithms.algorithm import Algorithm
from project.configs.datamodule import DataModuleConfig

logger = get_logger(__name__)
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@dataclass
class Config:
    """All the options required for a run. This dataclass acts as a schema for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    datamodule: DataModuleConfig
    """Configuration for the datamodule (dataset + transforms + dataloader creation)."""

    algorithm: Algorithm.HParams
    """The hyper-parameters of the algorithm to use."""

    network: Any
    """The network to use."""

    trainer: dict = field(default_factory=dict)
    """Keyword arguments for the Trainer constructor."""

    log_level: str = "info"

    # Random seed.
    seed: Optional[int] = field(default_factory=lambda: random.randint(0, int(1e5)))

    # Name for the experiment.
    name: str = "default"


Options = Config  # Alias for backward compatibility.

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

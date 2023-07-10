from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Literal, Optional, Callable, Any
import omegaconf
from torch import nn
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from project.algorithms.algorithm import Algorithm
from project.configs.datamodule import DataModuleConfig
from project.configs.network import ResNet18Config

logger = get_logger(__name__)
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@dataclass
class Config:
    """All the options required for a run. This dataclass acts as a schema for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    # Configuration for the datamodule (dataset + transforms + dataloader creation).
    datamodule: Any

    # The hyper-parameters of the algorithm to use.
    algorithm: Any

    # The network to use.
    network: Any

    # Keyword arguments for the Trainer constructor.
    trainer: dict = field(default_factory=dict)  # type: ignore

    # # Config(s) for the logger(s).
    # logger: Optional[dict] = field(default_factory=dict)  # type: ignore

    log_level: str = "info"

    # Random seed.
    seed: Optional[int] = None

    # Name for the experiment.
    name: str = "default"


Options = Config  # Alias for backward compatibility.

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

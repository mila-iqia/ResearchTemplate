from __future__ import annotations

from hydra.core.config_store import ConfigStore

from .config import Config
from .datamodule import (
    REPO_ROOTDIR,
    SLURM_JOB_ID,
    SLURM_TMPDIR,
    datamodule_store,
)
from .network import network_store

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
datamodule_store.add_to_hydra_store()
network_store.add_to_hydra_store()
# todo: move the algorithm_store.add_to_hydra_store() here?

__all__ = [
    "Config",
    "SLURM_TMPDIR",
    "SLURM_JOB_ID",
    "REPO_ROOTDIR",
]

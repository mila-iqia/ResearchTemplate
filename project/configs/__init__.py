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

# todo: look into using this instead:
# from hydra_zen import store

cs = ConfigStore.instance()
datamodule_store.add_to_hydra_store()
network_store.add_to_hydra_store()

__all__ = [
    "Config",
    "SLURM_TMPDIR",
    "SLURM_JOB_ID",
    "REPO_ROOTDIR",
]

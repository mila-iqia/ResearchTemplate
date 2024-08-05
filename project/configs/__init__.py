from __future__ import annotations

from hydra.core.config_store import ConfigStore

from project.configs.algorithm import register_algorithm_configs
from project.configs.config import Config
from project.configs.datamodule import datamodule_store
from project.configs.network import network_store
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID, SLURM_TMPDIR

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def add_configs_to_hydra_store(with_dynamic_configs: bool = True):
    """Adds all configs to the Hydra Config store."""
    datamodule_store.add_to_hydra_store()
    network_store.add_to_hydra_store()
    register_algorithm_configs(with_dynamic_configs=with_dynamic_configs)


# todo: move the algorithm_store.add_to_hydra_store() here?

__all__ = [
    "Config",
    "SLURM_TMPDIR",
    "SLURM_JOB_ID",
    "REPO_ROOTDIR",
]

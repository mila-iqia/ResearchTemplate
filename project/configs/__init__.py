"""All the configuration classes for the project."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from project.configs.algorithm.network import network_store
from project.configs.algorithm.optimizer import optimizers_store
from project.configs.config import Config
from project.configs.datamodule import datamodule_store
from project.utils.remote_launcher_plugin import RemoteSlurmQueueConf

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

OmegaConf.register_new_resolver("eval", eval)

ConfigStore.instance().store(
    group="hydra/launcher",
    name="remote_submitit_slurm",
    node=RemoteSlurmQueueConf,
    provider="Mila",
)


def add_configs_to_hydra_store():
    """Adds all configs to the Hydra Config store."""
    datamodule_store.add_to_hydra_store()
    network_store.add_to_hydra_store()
    optimizers_store.add_to_hydra_store()


# todo: move the algorithm_store.add_to_hydra_store() here?

__all__ = [
    "Config",
    # "SLURM_TMPDIR",
    # "SLURM_JOB_ID",
    # "REPO_ROOTDIR",
]

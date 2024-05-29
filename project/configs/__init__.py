from __future__ import annotations

from hydra.core.config_store import ConfigStore

from hydra_plugins.custom_launcher.custom_launcher import (
    CustomSlurmLauncher,
    CustomSlurmQueueConf,
)
from project.networks import FcNetConfig, ResNet18Config

from .config import Config
from .datamodule import (
    REPO_ROOTDIR,
    SLURM_JOB_ID,
    SLURM_TMPDIR,
    CIFAR10DataModuleConfig,
    DataModuleConfig,
    FashionMNISTDataModuleConfig,
    ImageNet32DataModuleConfig,
    INaturalistDataModuleConfig,
    MNISTDataModuleConfig,
    MovingMnistDataModuleConfig,
)

# todo: look into using this instead:
# from hydra_zen import store

cs = ConfigStore.instance()
cs.store(group="datamodule", name="base", node=DataModuleConfig)
cs.store(group="datamodule", name="cifar10", node=CIFAR10DataModuleConfig)
cs.store(group="datamodule", name="mnist", node=MNISTDataModuleConfig)
cs.store(group="datamodule", name="fashion_mnist", node=FashionMNISTDataModuleConfig)
cs.store(group="datamodule", name="imagenet32", node=ImageNet32DataModuleConfig)
cs.store(group="datamodule", name="inaturalist", node=INaturalistDataModuleConfig)
cs.store(group="datamodule", name="moving_mnist", node=MovingMnistDataModuleConfig)


cs.store(group="network", name="fcnet", node=FcNetConfig)
cs.store(group="network", name="resnet18", node=ResNet18Config)

__all__ = [
    "Config",
    "SLURM_TMPDIR",
    "SLURM_JOB_ID",
    "REPO_ROOTDIR",
    "CustomSlurmLauncher",
    "CustomSlurmQueueConf",
]

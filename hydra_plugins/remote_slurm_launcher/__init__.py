# Import this patch for https://github.com/mit-ll-responsible-ai/hydra-zen/issues/705 to make sure that it gets applied.
from ._remote_launcher_plugin import RemoteSlurmLauncher, RemoteSlurmQueueConf

__all__ = ["RemoteSlurmLauncher"]

from hydra.core.config_store import ConfigStore

ConfigStore.instance().store(
    group="hydra/launcher",
    name="remote_submitit_slurm",
    node=RemoteSlurmQueueConf(),
    provider="Mila",
)

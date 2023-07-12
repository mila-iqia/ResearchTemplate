from __future__ import annotations
from typing import Any, Sequence
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn
from hydra.types import HydraContext, TaskFunction
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher


OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)


@dataclass
class ClustomSlurmQueueConf(SlurmQueueConf):
    # _target_: str = (
    #     "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    # )

    _target_: str = (
        # "project.configs.resources.custom_launcher.CustomSlurmLauncher"
        "hydra_plugins.custom_launcher.custom_launcher.CustomSlurmLauncher"
    )

    # slurm partition to use on the cluster
    partition: str | None = None
    qos: str | None = None
    comment: str | None = None
    constraint: str | None = None
    exclude: str | None = None
    gres: str | None = None
    cpus_per_gpu: int | None = None
    # gpus_per_task: Optional[int] = None
    mem_per_gpu: str | None = None
    mem_per_cpu: str | None = None
    account: str | None = None

    # Following parameters are submitit specifics
    #
    # USR1 signal delay before timeout
    signal_delay_s: int = 120
    # Maximum number of retries on job timeout.
    # Change this only after you confirmed your code can handle re-submission
    # by properly resuming from the latest stored checkpoint.
    # check the following for more info on slurm_max_num_timeout
    # https://github.com/facebookincubator/submitit/blob/master/docs/checkpointing.md
    max_num_timeout: int = 0
    # Useful to add parameters which are not currently available in the plugin.
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    # Maximum number of jobs running in parallel
    array_parallelism: int = 256
    # A list of commands to run in sbatch before running srun
    setup: list[str] | None = None

    ### Modified:

    # Also support setting the GPU model, with `gpus_per_task: 'rtx8000:1'`
    gpus_per_task: int | str | None = None


class CustomSlurmLauncher(SlurmLauncher):
    def __init__(self, **params: Any) -> None:
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        self.config: DictConfig | None = None
        self.task_function: TaskFunction | None = None
        self.sweep_configs: TaskFunction | None = None
        self.hydra_context: HydraContext | None = None

    def setup(
        self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig
    ) -> None:
        return super().setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def __call__(
        self,
        sweep_overrides: list[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: dict[type, Singleton],
    ) -> JobReturn:
        return super().__call__(sweep_overrides, job_dir_key, job_num, job_id, singleton_state)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        return super().launch(job_overrides, initial_job_idx)

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        return super().checkpoint(*args, **kwargs)


ConfigStore.instance().store(
    group="hydra/launcher",
    name="custom_submitit_slurm",
    node=ClustomSlurmQueueConf(),
    provider="ResearchTemplate",
    # provider="submitit_launcher",
)

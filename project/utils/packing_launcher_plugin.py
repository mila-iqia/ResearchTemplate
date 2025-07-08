"""Submitit launcher 'patch' to support packing jobs with SLURM's `--ntasks-per-node`.

Originally implemented by Darshan Patil.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra.types import HydraContext, TaskFunction
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from omegaconf import DictConfig, OmegaConf
from submitit import SlurmExecutor
from submitit.core.job_environment import JobEnvironment

log = logging.getLogger(__name__)


# Made this a dataclass to avoid having an ugly default repr, but it causes issues with
# hydra-auto-schema because it tries to create a schema for everything here.
# @dataclasses.dataclass(init=False)
class PackedSubmititLauncher(BaseSubmititLauncher):
    _EXECUTOR: ClassVar[str] = "abstract"  # to be set in subclasses below.

    params: dict[str, Any]
    config: DictConfig | None = None
    task_function: TaskFunction | None = None
    sweep_configs: TaskFunction | None = None
    hydra_context: HydraContext | None = None
    executor: SlurmExecutor

    # Expanded the actual arguments to the slurm launcher.
    # This makes the code more explicit, and enabled autocomplete in IDEs in .py and .yaml files.
    def __init__(
        self,
        array_parallelism: int = 256,
        comment: str | None = None,
        constraint: str | None = None,
        cpus_per_gpu: int | None = None,
        cpus_per_task: int | None = None,
        dependency: str | None = None,
        exclude: str | None = None,
        exclusive: bool | None = None,
        gpus_per_node: int | str | None = None,
        gpus_per_task: int | str | None = None,
        gres: str | None = None,
        # job_name: str = "submitit",
        job_name: str = "submitit-${hydra.job.name}",
        mail_type: str | None = None,
        mail_user: str | None = None,
        mem: str | None = None,
        mem_per_cpu: str | None = None,
        mem_per_gpu: str | None = None,
        nodelist: str | None = None,
        nodes: int = 1,
        ntasks_per_node: int | None = None,
        num_gpus: int | None = None,
        partition: str | None = None,
        qos: str | None = None,
        setup: list[str] | None = None,
        signal_delay_s: int = 90,
        srun_args: list[str] | None = None,
        stderr_to_stdout: bool = True,  # changed!
        time: str | int = 5,
        use_srun: bool = True,
        wckey: str = "submitit",
        additional_parameters: dict | None = None,
        tasks_per_node: int | None = None,
        mem_gb: int | None = None,
        # Added:
        ntasks_per_gpu: int | None = None,
    ) -> None:
        setup = setup or []
        additional_parameters = additional_parameters or {}

        if mem_gb is not None:
            assert mem is None, "can't use both mem and mem_gb"
            mem = f"{mem_gb}GB"
        if tasks_per_node is not None:
            assert ntasks_per_node is None, "can't use both tasks_per_node and ntasks_per_node"
            ntasks_per_node = tasks_per_node
        if ntasks_per_node is not None:
            additional_parameters["ntasks-per-node"] = ntasks_per_node
        # Added:
        if ntasks_per_gpu is not None:
            assert gpus_per_task is None, "can't use both gpus_per_task and ntasks_per_gpu"
            gpus_per_task = ntasks_per_gpu
            additional_parameters["ntasks-per-gpu"] = ntasks_per_gpu

        super().__init__(
            array_parallelism=array_parallelism,
            comment=comment,
            constraint=constraint,
            cpus_per_gpu=cpus_per_gpu,
            cpus_per_task=cpus_per_task,
            dependency=dependency,
            exclude=exclude,
            exclusive=exclusive,
            gpus_per_node=gpus_per_node,
            gpus_per_task=gpus_per_task,
            gres=gres,
            job_name=job_name,
            mail_type=mail_type,
            mail_user=mail_user,
            mem=mem,
            mem_per_cpu=mem_per_cpu,
            mem_per_gpu=mem_per_gpu,
            nodelist=nodelist,
            nodes=nodes,
            num_gpus=num_gpus,
            partition=partition,
            qos=qos,
            setup=setup,
            signal_delay_s=signal_delay_s,
            srun_args=srun_args,
            stderr_to_stdout=stderr_to_stdout,
            time=time,
            use_srun=use_srun,
            wckey=wckey,
            additional_parameters=additional_parameters,
        )

    def launch_batch(
        self,
        sweep_overrides: list[list[str]],
        job_dir_key: list[str],
        job_num: list[int],
        job_id: list[str],
        singleton_state: list[dict[type, Singleton]],
    ) -> JobReturn:
        log.info(self.config)
        log.info(os.environ)

        task_id = JobEnvironment().global_rank
        return self(
            sweep_overrides[task_id],
            job_dir_key[task_id],
            job_num[task_id],
            job_id[task_id],
            singleton_state[task_id],
        )

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{f"{self._EXECUTOR}_{x}": y for x, y in params.items() if x in specific_init_keys}
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(f"Submitit '{self._EXECUTOR}' sweep output dir : {self.config.hydra.sweep.dir}")
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: list[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        jobs = executor.map_array(
            self.launch_batch,
            *list(batch(jps, self.params["tasks_per_node"]) for jps in zip(*job_params)),
        )
        # Unpack all results from all tasks in each job.
        return [res for j in jobs for res in j.results()]


T = TypeVar("T")


def batch(x: Sequence[T], bs: int) -> list[Sequence[T]]:
    return [x[i : i + bs] for i in range(0, len(x), bs)]


class LocalLauncher(PackedSubmititLauncher):
    _EXECUTOR = "local"


class SlurmLauncher(PackedSubmititLauncher):
    _EXECUTOR = "slurm"

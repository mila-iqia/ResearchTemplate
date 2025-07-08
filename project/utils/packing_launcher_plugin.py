"""Submitit launcher 'patch' to support packing jobs with SLURM's `--ntasks-per-node`.

Originally implemented by Darshan Patil.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import submitit
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra.types import HydraContext, TaskFunction
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from omegaconf import DictConfig, OmegaConf
from submitit import SlurmExecutor
from submitit.core.job_environment import JobEnvironment

logger = logging.getLogger(__name__)


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
        # TODO: clean up this mess with arbitrary kwargs and prefixes by just passing the executor.
        # executor: Callable[..., submitit.Executor] = submitit.SlurmExecutor,
        *,
        # Added:
        ntasks_per_gpu: int | None = None,
        # NOTE: this **kwargs a mix of things: The constructor arguments for the executor, + some
        # arguments passed to executor.update_parameters
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        additional_parameters = self.params.setdefault("additional_parameters", {})
        assert isinstance(additional_parameters, dict)

        # Added:
        if ntasks_per_gpu is not None:
            additional_parameters["ntasks_per_gpu"] = ntasks_per_gpu
            if self.params["ntasks_per_node"] is None:
                # If we only set ntasks_per_gpu, then infer that ntasks_per_node is the same.
                self.params["ntasks_per_node"] = ntasks_per_gpu

    def launch_batch(
        self,
        sweep_overrides: list[list[str]],
        job_dir_key: list[str],
        job_num: list[int],
        job_id: list[str],
        singleton_state: list[dict[type, Singleton]],
    ) -> JobReturn:
        logger.info(self.config)
        logger.info(os.environ)

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
        additional_parameters = self.params.get("additional_parameters", {})
        assert isinstance(additional_parameters, dict)
        num_tasks: int = 1
        if "task_per_node" in self.params:
            num_tasks = int(self.params["tasks_per_node"])
        elif "ntask_per_node" in self.params:
            num_tasks = int(self.params["ntasks_per_node"])
        elif "ntasks_per_gpu" in additional_parameters:
            num_tasks = int(additional_parameters["ntasks_per_gpu"])

        if num_tasks == 1:
            # DO the same as the original submitit launcher.
            logger.info("Not using job packing, using the default `launch` method.")
            return super().launch(job_overrides, initial_job_idx)

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        if "max_num_timeout" in params:
            init_params[f"{self._EXECUTOR}_max_num_timeout"] = params["max_num_timeout"]
        init_keys = {"max_num_timeout", "submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        logger.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : {self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: list[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            logger.info(f"\t#{idx} : {lst}")
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
            *list(batch(jps, num_tasks) for jps in zip(*job_params)),
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

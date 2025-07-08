"""Submitit launcher 'patch' to support packing jobs with SLURM's `--ntasks-per-node`.

Originally implemented by Darshan Patil.
"""

import logging
import os
import warnings
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
class _PackedSubmititLauncher(BaseSubmititLauncher):
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
        # NOTE: this **kwargs a mix of things: The constructor arguments for the executor, + some
        # arguments passed to executor.update_parameters
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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
        # TODO: For Lightning, we need to patch `rank_zero_only` so that logging happens in all
        # tasks.
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
        ntasks_per_node: int | None = self.params.get("tasks_per_node")

        # TODO: use the ntasks_per_gpu sbatch flag instead!
        ntasks_per_gpu: int | None = additional_parameters.get("ntasks_per_gpu")
        if ntasks_per_gpu is None:
            ntasks_per_gpu = 1
        if ntasks_per_gpu > 1 and ntasks_per_node not in (None, 1):
            warnings.warn(
                RuntimeWarning(
                    "Using `tasks_per_node` > 1 with `tasks_per_gpu` > 1 is potentially problematic. "
                )
            )

        if ntasks_per_gpu == 1:
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
            *list(batch(jps, ntasks_per_gpu) for jps in zip(*job_params)),
        )
        # Unpack all results from all tasks in each job.
        return [res for j in jobs for res in j.results()]


T = TypeVar("T")


def batch(x: Sequence[T], bs: int) -> list[Sequence[T]]:
    return [x[i : i + bs] for i in range(0, len(x), bs)]


class PackedLocalLauncher(_PackedSubmititLauncher):
    _EXECUTOR = "local"


class PackedSlurmLauncher(_PackedSubmititLauncher):
    _EXECUTOR = "slurm"

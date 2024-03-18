from __future__ import annotations

import dataclasses
import enum
import os
import subprocess
import warnings
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, NamedTuple, Sequence, TypeVar, Optional, Union

import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn
from hydra.types import HydraContext, TaskFunction
from hydra_zen import hydrated_dataclass
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeGuard

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    SlurmLauncher,
    filter_overrides,
)
from project.utils.hydra_utils import interpolated_field

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


logger = get_logger(__name__)

OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)


def keys_are(d: dict[Any, V], key_type: type[K]) -> TypeGuard[dict[K, V]]:
    return all(isinstance(k, key_type) for k, _ in d.items())


def values_are(d: dict[K, Any], value_type: type[V]) -> TypeGuard[dict[K, V]]:
    return all(isinstance(v, value_type) for _, v in d.items())


class GpuModel(enum.Enum):
    a100_10gb = "1g.10gb"
    a100_20gb = "2g.20gb"
    a100_40gb = "3g.40gb"
    a100 = "a100"
    a100l = "a100l"
    a6000 = "a6000"
    rtx8000 = "rtx8000"
    v100 = "v100"


gpu_memory_gb = {
    "1g.10gb": 10,
    "2g.20gb": 20,
    "3g.40gb": 40,
    "a100": 40,
    "a100l": 80,
    "a6000": 48,
    "rtx8000": 48,
    "v100": 16,
}


class AvailTotal(NamedTuple):
    avail: int
    total: int


def savail() -> dict[str, AvailTotal]:
    """Gets the output of the `savail` command in a Python dictionary.

    ```
     GPU               Avail / Total
    ===============================
    1g.10gb              38 / 40
    2g.20gb              59 / 60
    3g.40gb              39 / 40
    a100                  0 / 16
    a100l                 3 / 56
    a6000                 1 / 8
    rtx8000              156 / 384
    v100                 10 / 50
    ```
    """
    savail_output = subprocess.check_output(["savail"]).decode("utf-8")
    lines = [line.strip() for line in savail_output.splitlines()[2:]]
    return {
        gpu_type: AvailTotal(int(avail), int(total))
        for gpu_type, avail, _, total in [line.split() for line in lines]
    }


@dataclass
class CustomSlurmQueueConf(SlurmQueueConf):
    _target_: str = (
        "hydra_plugins.custom_launcher.custom_launcher.CustomSlurmLauncher"
        # "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    )

    partition: Optional[str] = None
    """Slurm partition to use on the cluster."""
    qos: Optional[str] = None
    comment: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    gres: Optional[str] = None
    cpus_per_gpu: Optional[int] = None
    gpus_per_task: Optional[Union[int, str]] = None
    mem_per_gpu: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    account: Optional[str] = None

    ################################
    ### Submitit-specific fields ###
    ################################

    signal_delay_s: int = 120
    """USR1 signal delay before timeout."""

    max_num_timeout: int = 0
    """Maximum number of retries on job timeout.

    Change this only after you confirmed your code can handle re-submission by properly resuming
    from the latest stored checkpoint. check the following for more info on slurm_max_num_timeout
    https://github.com/facebookincubator/submitit/blob/master/docs/checkpointing.md
    """

    # Useful to add parameters which are not currently available in the plugin.
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    # Maximum number of jobs running in parallel
    array_parallelism: int = 256
    # A list of commands to run in sbatch before running srun
    setup: list[str] = field(default_factory=list)

    ########################
    ### Added attributes ###
    ########################

    # ntasks_per_gpu: int | None = None  # TODO: Not yet supported by submitit

    # max_vram_usage_gb: int | None = None  # FIXME: remove default
    """The maximum VRAM usage of your job.

    Knowing this in advance can be very useful, since it allows the launcher to automatically pack
    multiple runs in parallel within a single job.
    """
    # gpu_type: GpuModel | str | None = None  # FIXME: remove default

    # parallel_runs_per_job: int | None = None

    # gpus: int | str | None = None

    srun_args: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    """Configuration for the resources of a training run.

    IDEA: This gets "translated" into SLURM requirements. Probably not a good idea though.
    """

    # nodes: int = interpolated_field("${trainer.nodes}", 1)
    # """ Number of nodes required. """

    cpus: Optional[int] = interpolated_field("${datamodule.num_workers}", None)
    """Number of CPU cores required."""

    ram_gb: Optional[int] = 4
    """Amount of CPU RAM required (for a single run)."""

    gpus: Optional[int] = interpolated_field("${trainer.devices}", None)
    """Number of GPUs required."""

    vram_gb: Optional[int] = None
    """Amount of GPU VRAM required."""

    gpu_type: Optional[Union[GpuModel, str]] = None
    """Type of GPU required for this job."""

    share_cpus_between_runs: bool = True
    """Whether to allow different runs within the same job to share CPU cores."""

    parallel_runs_per_job: Optional[int] = None
    """How many distinct runs (~tasks) to execute in parallel within a single job.

    When `max_vram_usage_gb` is set, and a gpu model has been selected either in "gpu_model",
    "gres" or "gpus", this number is automatically set to `max_vram_usage_gb // gpu_memory_gb`.

    Make sure that you use the `SLURM_PROCID` somehow so that each run produces different results.
    One good way of doing that is by using `SLURM_PROCID` as part of the random seed.
    """


def translate_into_slurm_params(
    job_config: JobConfig, manual_args: CustomSlurmQueueConf
) -> CustomSlurmQueueConf:
    assert job_config.parallel_runs_per_job is not None  # FIXME
    assert job_config.ram_gb is not None  # FIXME
    gpu_type = (
        job_config.gpu_type.value
        if isinstance(job_config.gpu_type, GpuModel)
        else job_config.gpu_type
    )
    assert gpu_type is not None  # FIXME

    srun_args = manual_args.srun_args.copy()
    additional_parameters = manual_args.additional_parameters.copy()

    if job_config.share_cpus_between_runs and "--overcommit" not in srun_args:
        srun_args.append("--overcommit")
        # Make the cpus_per_task be interpreted as the number of cpus per node.
        additional_parameters["overcommit"] = True

    setup: list[str] = manual_args.setup.copy()

    return dataclasses.replace(
        manual_args,
        gres=f"gpu:{gpu_type}:{job_config.gpus}",
        cpus_per_task=job_config.cpus,
        mem_gb=job_config.ram_gb * job_config.parallel_runs_per_job,
        tasks_per_node=job_config.parallel_runs_per_job,
        srun_args=srun_args,
        # setup=[]
        additional_parameters=additional_parameters,
        setup=setup,
        # # Other things to pass to `sbatch`:
        # additional_parameters:
        #   time: 1-00:00:00  # maximum wall time allocated for the job (D-HH:MM:SS)
        #   requeue: True
        #   overcommit: True  # Make the cpus_per_task above interpreted as the number of cpus per
        # node.
    )


class CustomSlurmLauncher(SlurmLauncher):
    """# TODOs / IDEAs:

    - Ask @obilaniu for other useful srun / sbatch args that we should set.
    - Find out the max VRAM usage by reading the nvidia-smi part of the job epilog! (has the max
      vram usage over the course of the entire job).
    - Do a quick interactive menu before launching the sweep, to ask what kind of GPU to target
      (also showing the output of savail)
    - (detail): Infer the GPU model from the "gpus" or "gres" flags if not passed.
    """

    def __init__(self, job_config: JobConfig, **params) -> None:
        super().__init__(**params)
        self.params: dict[str, Any]
        self.job_config = job_config

    def adjust_job_config(self, job_config: JobConfig) -> JobConfig:
        gpu_type = job_config.gpu_type
        required_vram_gb = job_config.vram_gb
        gpus: int | None = job_config.gpus
        parallel_runs_per_job = job_config.parallel_runs_per_job

        if (
            gpu_type is not None
            and required_vram_gb is not None
            and parallel_runs_per_job is not None
        ):
            logger.debug("Job config looks good, not adjusting anything.")
            return job_config

        if gpu_type is None or not isinstance(gpus, int):
            for flag in ["gres", "gpus", "gpus_per_task"]:
                value = self.params[flag]
                if isinstance(value, str) and value.count(":") == 1:
                    # TODO: Might not always be true, IIRC there could also be a constraint or
                    # VRAM or something.
                    model, _, gpus_str = value.partition(":")
                    gpu_type = GpuModel[model]
                    gpus = int(gpus_str)
                    logger.debug(
                        f"Inferred {gpu_type=} and num_gpus of {gpus} from {flag}={value}"
                    )
                    break

        if gpu_type is None:
            gpus_available = savail()
            compatible_gpus = gpus_available
            if job_config.vram_gb:
                compatible_gpus = {
                    k if job_config.vram_gb < gpu_memory_gb[k] else k for k in gpus_available
                }
            most_widely_available_gpu = max(compatible_gpus, key=lambda k: gpus_available[k][0])

            warnings.warn(
                RuntimeWarning(
                    f"You didn't specify a GPU model to use! This means that we can't "
                    f"automatically pack multiple runs per job for you.\n"
                    f"We suggest you use `gpu_model={'|'.join(gpu_memory_gb)}`."
                    f"Here are some of the currently available GPU models: \n"
                    f"{gpus_available}\n"
                    f"Of these, the most widely available GPU is {most_widely_available_gpu}.\n"
                )
            )

        if required_vram_gb is not None and gpu_type and parallel_runs_per_job is None:
            # TODO: Also take the RAM into account here.
            gpu_vram = gpu_memory_gb[
                gpu_type.value if isinstance(gpu_type, GpuModel) else gpu_type
            ]
            parallel_runs_per_job = gpu_vram // required_vram_gb
            logger.info(
                f"Automatically set the number of parallel runs per job to {parallel_runs_per_job}"
            )

        return dataclasses.replace(
            job_config,
            vram_gb=required_vram_gb,
            gpu_type=gpu_type,
            parallel_runs_per_job=parallel_runs_per_job,
        )

    def setup(
        self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig
    ) -> None:
        super().setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def __call__(
        self,
        sweep_overrides: list[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: dict[type, Singleton],
    ) -> JobReturn:
        return super().__call__(sweep_overrides, job_dir_key, job_num, job_id, singleton_state)

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        return super().checkpoint(*args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import submitit
        from submitit.core.utils import DelayedSubmission

        assert self.config is not None

        job_config = self.adjust_job_config(self.job_config)

        parallel_runs_per_job: int = job_config.parallel_runs_per_job or 1

        manual_slurm_params = CustomSlurmQueueConf(**self.params)

        run_slurm_params: CustomSlurmQueueConf = translate_into_slurm_params(
            job_config, manual_slurm_params
        )

        if job_config.parallel_runs_per_job and job_config.parallel_runs_per_job > 1:
            logger.info(
                f"Packing {job_config.parallel_runs_per_job} runs into each job by increasing "
                f"ntasks_per_node to {run_slurm_params.tasks_per_node}."
            )

        # build executor
        num_jobs = len(job_overrides)
        assert num_jobs > 0

        params = dataclasses.asdict(run_slurm_params)

        # TODO: Need to get rid of the _target_ flags.
        params = _remove_hydra_fields_recursive(params)
        assert isinstance(params, dict)
        assert keys_are(params, str)

        job_folder = Path(params.pop("submitit_folder"))
        max_num_timeout: int = params.pop("max_num_timeout")

        executor = submitit.SlurmExecutor(
            folder=job_folder,
            max_num_timeout=max_num_timeout,
        )

        srun_args: list[str] = params.setdefault("srun_args", [])
        # TODO: Add an environment variable to give the run index (which would be different from
        # $SLURM_PROCID when using >1 gpus per job.)
        if job_config.parallel_runs_per_job:
            srun_args.append(f"--export=ALL,TASKS_PER_RUN={job_config.parallel_runs_per_job}")

        params = executor._convert_parameters(params)
        executor.update_parameters(**params)

        # NOTE: To see the file that would be generated, do:
        # print(executor._make_submission_file_text("foobar", "123123"))

        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Submitit sweep output dir : " f"{sweep_dir}")

        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: list[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            overrides_list = " ".join(filter_overrides(overrides))
            logger.info(f"\t#{idx} : {overrides_list}")
            # TODO: Could perhaps use a TypedDict or a NamedTuple, to show that this is the args
            # that are passed to self.__call__.
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",  # wtf?
                    Singleton.get_state(),
                )
            )

        # IDEA: Inject a few things before, during, and after running the sweep:
        # jobs = executor.map_array(self, *zip(*job_params))
        # NOTE: Just inspecting things, this `map_array` is expanded into:

        fn = self
        iterable = zip(*job_params)

        submissions = [DelayedSubmission(fn, *args) for args in zip(*iterable)]
        if len(submissions) == 0:
            warnings.warn("Received an empty job array")
            return []
        jobs = executor._internal_process_submissions(submissions)

        # TODO: It discards the results of other tasks than the first for each job!!
        job_task_results: list[list[JobReturn]] = [j.results() for j in jobs]
        # assert False, [len(results_per_task[0]), results_per_task[0][0]]

        # TODO: Hacky: Set the result to the average across seeds? Or return a list as the result?
        # Important: we shouldn't necessarily assume that we're using the Orion sweeper here...
        # for task_results in job_task_results:
        #     task_results[0].return_value = [jr.return_value for jr in task_results]

        # FIXME: Trying to return more results than inputs.
        # return sum(job_task_results, [])
        if parallel_runs_per_job > 1:
            warnings.warn(
                RuntimeWarning(
                    "NOTE: Running multiple seeds in each job, but only able to report the first "
                    "result.."
                )
            )

        average_results = []
        for task_results in job_task_results:
            return_values: list[int | float] = []
            for result in task_results:
                if isinstance(result.return_value, (int, float)):
                    return_values.append(result.return_value)
            if not return_values:
                break
            if np.isnan(return_values).any():
                # TODO: Do something about it?
                continue
            average_results.append(np.mean(return_values))

        # TODO: Perhaps turn this behaviour on-off with a flag? Because we might be using
        # multi-gpu jobs or something, so perhaps we don't want to do this?
        if len(average_results) == len(job_task_results):
            return average_results

        # Only return one JobReturn per job. (default behaviour of the launcher of Hydra).
        return [task_results[0] for task_results in job_task_results]

        # return [j.results()[0] for j in jobs]


@hydrated_dataclass(target=CustomSlurmLauncher, hydra_convert="object")
class CustomLauncherConfig(CustomSlurmQueueConf):
    _target_: str = CustomSlurmLauncher.__module__ + "." + CustomSlurmLauncher.__qualname__

    job_config: JobConfig = field(default_factory=JobConfig)
    # slurm: CustomSlurmQueueConf = field(default_factory=CustomSlurmQueueConf)

    srun_args: list[str] = field(default_factory=list)
    """Additional arguments to pass to `srun` in `srun (...) <here> (...) python (...)`"""

    stderr_to_stdout: bool = True
    """Whether to use the same file for stderr and stdout."""

    # ntasks_per_gpu: int | None = None  # TODO: Not yet supported by submitit. Could perhaps be
    # used as the "nruns_per_job" parameter.


def _remove_hydra_fields_recursive(d: dict[str, V]) -> dict[str, V]:
    return {
        k: _remove_hydra_fields_recursive(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if not (k.startswith("_") and k.endswith("_"))
    }


ConfigStore.instance().store(
    group="hydra/launcher",
    name="custom_submitit_slurm",
    node=CustomLauncherConfig,
    # provider="ResearchTemplate",
    # provider="submitit_launcher",
)

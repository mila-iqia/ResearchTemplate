from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Any, NamedTuple, Sequence
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn
import os
import warnings
from hydra.types import HydraContext, TaskFunction
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    SlurmLauncher,
    filter_overrides,
)
import enum
from logging import getLogger as get_logger


logger = get_logger(__name__)

OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)


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
class ClustomSlurmQueueConf(SlurmQueueConf):
    _target_: str = (
        "hydra_plugins.custom_launcher.custom_launcher.CustomSlurmLauncher"
        # "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
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

    ########################
    ### Added attributes ###
    ########################

    # Also support setting the GPU model, with `gpus_per_task: 'rtx8000:1'`
    gpus_per_task: int | str | None = None

    ntasks_per_gpu: int | None = None  # TODO: Not yet supported by submitit

    max_vram_usage_gb: int | None = None  # FIXME: remove default
    """The maximum VRAM usage of your job.

    Knowing this in advance can be very useful, since it allows the launcher to automatically pack
    multiple runs in parallel within a single job.
    """
    gpu_model: GpuModel | str | None = None  # FIXME: remove default

    parallel_runs_per_job: int | None = None
    """How many distinct runs to execute in parallel within a single job.

    When `max_vram_usage_gb` is set, and a gpu model has been selected either in "gpu_model",
    "gres" or "gpus", this number is automatically set to `max_vram_usage_gb // gpu_memory_gb`.

    Make sure that you use the `SLURM_PROCID` somehow so that each run produces different results.
    One good way of doing that is by using `SLURM_PROCID` as part of the random seed.
    """

    gpus: int | str | None = None

    srun_args: list[str] | None = None


class CustomSlurmLauncher(SlurmLauncher):
    """# TODOs / IDEAs:

    - Ask @obilaniu for other useful srun / sbatch args that we should set.
    - Find out the max VRAM usage by reading the nvidia-smi part of the job epilog! (has the max
      vram usage over the course of the entire job).
    - Do a quick interactive menu before launching the sweep, to ask what kind of GPU to target
      (also showing the output of savail)
    - (detail): Infer the GPU model from the "gpus" or "gres" flags if not passed.
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # self.params_obj = ClustomSlurmQueueConf(**self.params)
        gpus: int | str | None = self.params.pop("gpus", None)
        gpu_model: GpuModel | str | None = self.params.pop("gpu_model", None)
        parallel_runs_per_job: int | None = self.params.pop("parallel_runs_per_job", None)
        max_vram_usage_gb: int | None = self.params.pop("max_vram_usage_gb", None)

        if gpu_model is None:
            if (
                (gres := self.params["gres"])
                and ":" in gres
                and (model := gres.partition(":")[0]) in gpu_memory_gb.keys()
            ):
                gpu_model = GpuModel[model]

            if (
                (gpus := self.params["gpus"])
                and isinstance(gpus, str)
                and ":" in gpus
                and (model := gpus.partition(":")[0]) in gpu_memory_gb.keys()
            ):
                gpu_model = GpuModel[model]
        if gpu_model is None:
            warnings.warn(
                RuntimeWarning(
                    f"You didn't specify a GPU model to use! This means that we can't "
                    f"automatically pack multiple runs per job for you.\n"
                    f"We suggest you use `gpu_model={'|'.join(gpu_memory_gb)}`."
                    f"Here are some of the currently available GPU models: \n"
                    f"{savail()}"
                )
            )

        if max_vram_usage_gb is not None and gpu_model and parallel_runs_per_job is None:
            gpu_vram = gpu_memory_gb[
                gpu_model.value if isinstance(gpu_model, GpuModel) else gpu_model
            ]
            parallel_runs_per_job = gpu_vram // max_vram_usage_gb
            logger.info(
                f"Automatically set the number of parallel runs per job to {parallel_runs_per_job}"
            )
        self.max_vram_usage_gb = max_vram_usage_gb
        self.gpu_model = gpu_model
        self.parallel_runs_per_job = parallel_runs_per_job
        self.gpus = gpus

        # assert False, f"About to stack {num_stacked_runs} runs"
        # self.config: DictConfig | None = None
        # self.task_function: TaskFunction | None = None
        # self.sweep_configs: TaskFunction | None = None
        # self.hydra_context: HydraContext | None = None

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

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        return super().checkpoint(*args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.config is not None
        # build executor
        num_jobs = len(job_overrides)
        assert num_jobs > 0

        params = self.params.copy()

        executor = submitit.SlurmExecutor(
            folder=params.pop("submitit_folder"),
            max_num_timeout=params.pop("max_num_timeout"),
        )
        eq_dict = executor._equivalence_dict()
        params = {eq_dict.get(k, k): v for k, v in params.items()}

        # Enable multiple tasks sharing the same CPU.
        # TODO: Check that this makes sense!
        srun_args: list[str] = params.setdefault("srun_args", [])
        if "--overcommit" not in srun_args:
            srun_args.append("--overcommit")

        # Additional flags for sbatch.
        additional_parameters: dict[str, Any] = params.setdefault("additional_parameters", {})

        # TODO: Need to decide how we want to "pack" the multiple jobs: Either with ntasks_per_gpu
        # or #ntasks_per_node. PyTorch-Lightning complains if we don't set ntasks_per_node, which
        # is quite frustrating.

        # FIXME: Opting for `ntasks_per_node` approach for now.
        ntasks_per_gpu: int | None = params.pop("ntasks_per_gpu", None)
        # ntasks_per_gpu = ntasks_per_gpu or self.parallel_runs_per_job  # Option 1

        if ntasks_per_gpu:
            additional_parameters.update({"ntasks_per_gpu": ntasks_per_gpu})

        ntasks_per_node = params.pop("ntasks_per_node", None) or 1
        assert isinstance(ntasks_per_node, int)
        if self.parallel_runs_per_job:
            prev = ntasks_per_node
            ntasks_per_node *= self.parallel_runs_per_job  # Option 2
            logger.info(
                f"Packing {self.parallel_runs_per_job} runs into each job by increasing "
                f"ntasks_per_node from {prev} to {ntasks_per_node}"
            )
        params["ntasks_per_node"] = ntasks_per_node

        if self.gpus is not None:
            additional_parameters.update({"gpus": self.gpus})

        if (mem := params["mem"]) and isinstance(mem, int) or not mem.endswith("G"):
            params["mem"] = f"{mem}G"

        params["stderr_to_stdout"] = True

        if ntasks_per_gpu and "ntasks_per_node" in params:
            logger.debug(
                f"Removing `ntasks_per_node={ntasks_per_node}` from sbatch parameters "
                f"since we'll be using ntasks_per_gpu to pack multiple runs in each job."
            )

        # TODO: Add an environment variable to give the run index (which would be different from
        # $SLURM_PROCID when using >1 gpus per job.)
        # srun_args.append("--export=ALL,RUN_INDEX=%t")

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
            lst = " ".join(filter_overrides(overrides))
            logger.info(f"\t#{idx} : {lst}")
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
        # This is expanded into:
        fn = self
        iterable = zip(*job_params)
        from submitit.core.utils import DelayedSubmission

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
        # BUG: This seems to make the Orion sweeper launch multiple array jobs instead of just one!
        # Very interesting!
        # return sum(job_task_results, [])

        # Only return one JobReturn per job. Not sure if things would break otherwise..
        return [task_results[0] for task_results in job_task_results]

        # return [j.results()[0] for j in jobs]


ConfigStore.instance().store(
    group="hydra/launcher",
    name="custom_submitit_slurm",
    node=ClustomSlurmQueueConf(),
    # provider="ResearchTemplate",
    # provider="submitit_launcher",
)

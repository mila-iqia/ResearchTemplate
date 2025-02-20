# IDEA: Replacement for the --multirun option of Hydra.
# from hydra.main import   # noqa

import argparse
import collections
import contextlib
import dataclasses
import datetime
import functools
import logging
import os
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeGuard, TypeVar, overload

import hydra
import hydra_zen
import rich.logging
import simple_parsing
import submitit
import submitit.core.utils
import submitit.slurm
import submitit.slurm.slurm
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra._internal.utils import _get_completion_help
from hydra.core.override_parser.overrides_parser import OverridesParser
from remote_slurm_executor import RemoteSlurmExecutor
from remote_slurm_executor.slurm_remote import RemoteSlurmJob
from submitit.slurm.slurm import SlurmJob, _make_sbatch_string

from project.conftest import command_line_overrides  # noqa

logger = logging.getLogger(__name__)


def split_arguments_for_each_run(multirun_overrides: list[str]) -> list[list[str]]:
    overrides_objects = OverridesParser.create().parse_overrides(multirun_overrides)
    batches = BasicSweeper.split_arguments(overrides_objects, max_batch_size=None)
    assert len(batches) == 1  # all jobs in a single batch, since `max_batch_size=None`.
    return batches[0]


SCRATCH = Path(os.environ["SCRATCH"])


@dataclasses.dataclass
class HydraArgs:
    overrides: list[str] = simple_parsing.field(positional=True)
    """Any key=value arguments to override config values (use dots for.nested=overrides)"""

    help: bool = simple_parsing.field(alias=["-h", "--help"], default=False)
    """Application's help."""

    hydra_help: bool = simple_parsing.field(default=False)
    """Hydra's help."""

    version: bool = simple_parsing.field(
        default=False, action="version", version=f"Hydra {hydra.__version__}"
    )
    """Show Hydra's version and exit."""

    cfg: Literal["job", "hydra", "all"] | None = simple_parsing.field(
        alias=["-c", "--cfg"], default=None
    )
    """Show config instead of running [job|hydra|all]"""

    resolve: bool = False
    """Used in conjunction with --cfg, resolve config interpolations before printing."""

    package: str | None = simple_parsing.field(alias=["-p", "--package"], default=None)
    """Config package to show."""

    run: bool = simple_parsing.field(alias=["-r", "--run"], default=False)
    """Run a job."""

    multirun: bool = simple_parsing.field(alias=["-m", "--multirun"], default=False)
    """Run multiple jobs with the configured launcher and sweeper."""

    # defer building the completion help string until we actually need to render it
    class LazyCompletionHelp:
        # def __add__(self, other):
        #     return f"{self}{other}"

        def __repr__(self) -> str:
            return f"Install or Uninstall shell completion:\n{_get_completion_help()}"

    shell_completion: str = simple_parsing.field(
        alias=["-sc", "--shell-completion"], default=argparse.SUPPRESS, help=LazyCompletionHelp()
    )

    config_path: Path | None = simple_parsing.field(alias=["-cp", "--config-path"], default=None)
    """Overrides the config_path specified in hydra.main().

    The config_path is absolute or relative to the Python file declaring @hydra.main()
    """

    config_name: str | None = simple_parsing.field(alias=["-cn", "--config-name"], default=None)
    """Overrides the config_name specified in hydra.main()"""

    config_dir: Path | None = simple_parsing.field(alias=["-cd", "--config-dir"], default=None)
    """Adds an additional config dir to the config search path."""

    experimental_rerun: bool = False
    """Rerun a job from a previous config pickle."""

    info_choices = [
        "all",
        "config",
        "defaults",
        "defaults-tree",
        "plugins",
        "searchpath",
    ]
    info: Literal["all", "config", "defaults", "defaults-tree", "plugins", "searchpath"] | None = (
        simple_parsing.field(
            alias=["-i", "--info"],
            default="all",
            nargs="?",
            choices=info_choices,
            const="all",
        )
    )
    """Print Hydra information."""


_SbatchArgs = hydra_zen.kwargs_of(
    _make_sbatch_string, zen_exclude=("command", "folder", "map_count")
)


@dataclasses.dataclass
class SbatchArgs(_SbatchArgs):
    time: int | str = 10
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 4
    gpus_per_task: int | str | None = 1
    # ntasks_per_gpu: int | None = None  # todo: support this!
    mem: str = "16G"
    stderr_to_stdout: bool = True


@hydra_zen.hydrated_dataclass(submitit.SlurmExecutor)
class SlurmExecutorArgs:
    folder: Path = dataclasses.field(
        default=SCRATCH / "logs" / datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    )
    max_num_timeout: int = 0

    # probably unused, actually:
    python: str = "uv run --all-extras --frozen python"


@hydra_zen.hydrated_dataclass(RemoteSlurmExecutor)
class RemoteSlurmExecutorArgs:
    folder: str | Path = dataclasses.field(
        default=Path("logs") / datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    )
    max_num_timeout: int = 0

    # probably unused, actually:
    python: str = "uv run --all-extras --frozen python"


@dataclasses.dataclass(frozen=True)
class Dependency:
    job: SlurmJob | list[SlurmJob]
    type: Literal["afterok", "afterany", "afternotok", "afterburstbuffer", "aftercorr"]

    def to_str(self):
        job_ids = (
            [job.job_id for job in self.job] if isinstance(self.job, list) else [self.job.job_id]
        )
        return f"{self.type}:{':'.join(job_ids)}"


@overload
def launch(
    job_commands: list[str],
    executor_args: SlurmExecutorArgs,
    resources: SbatchArgs,
    cluster: Literal["current"] = "current",
    dependency: Literal["singleton"] | Dependency | None = ...,
) -> SlurmJob[str]: ...


@overload
def launch(
    job_commands: list[str],
    executor_args: SlurmExecutorArgs,
    resources: SbatchArgs,
    cluster: Literal["mila", "narval", "beluga", "cedar", "graham"] | str,
    dependency: Literal["singleton"] | Dependency | None = None,
) -> RemoteSlurmJob[str]: ...


def launch(
    job_commands: list[str],
    executor_args: SlurmExecutorArgs,
    resources: SbatchArgs,
    cluster: str = "current",
    dependency: Literal["singleton"] | Dependency | None = None,
) -> SlurmJob[str] | RemoteSlurmJob[str]:
    job = _launch(
        job_commands,
        executor_args=executor_args,
        resources=resources,
        cluster=cluster,
        dependency=dependency,
    )
    assert not isinstance(job, list)
    return job


def launch_array(
    job_commands: list[list[str]],
    *,
    executor_args: SlurmExecutorArgs,
    resources: SbatchArgs,
    cluster: str = "current",
    dependency: Dependency | None = None,
) -> list[SlurmJob[str]] | list[RemoteSlurmJob[str]]:
    jobs = _launch(
        job_commands,
        executor_args=executor_args,
        resources=resources,
        cluster=cluster,
        dependency=dependency,
    )
    assert isinstance(jobs, list)
    return jobs


def _launch(
    job_commands: list[str] | list[list[str]],
    executor_args: SlurmExecutorArgs,
    resources: SbatchArgs,
    cluster: str = "current",
    dependency: Literal["singleton"] | Dependency | None = None,
) -> SlurmJob[str] | RemoteSlurmJob[str] | list[SlurmJob[str]] | list[RemoteSlurmJob[str]]:
    """Launches a single job or a job array."""
    with (
        submitit.helpers.RsyncSnapshot(snapshot_dir=executor_args.folder / "code", root_dir=None)
        if cluster == "current"
        else contextlib.nullcontext()
    ):
        executor: submitit.SlurmExecutor | RemoteSlurmExecutor
        if cluster == "current":
            executor = hydra_zen.instantiate(executor_args)
        else:
            executor = hydra_zen.instantiate(executor_args, cluster_hostname=cluster)
        executor.update_parameters(
            **{k: v for k, v in dataclasses.asdict(resources).items() if k != "_target_"}
        )

        if dependency:
            additional_parameters = (executor.parameters.get("additional_parameters") or {}).copy()
            additional_parameters["kill-on-invalid-dep"] = "yes"
            executor.update_parameters(
                **{
                    "dependency": dependency
                    if isinstance(dependency, str)
                    else dependency.to_str(),
                    "additional_parameters": additional_parameters,
                }
            )

        offline = False
        if isinstance(executor, RemoteSlurmExecutor):
            offline = not executor.internet_access_on_compute_nodes

        ## Launch the test job
        logging.info(f"Working directory: {executor.folder}")

        # idea: Could run tests specific to that particular config (that use some of the job_args?)
        uv_command = ["uv", "run", "--all-extras", "--frozen"]
        if offline:
            # todo: also do a `uv sync` on the login node before submitting the job so we have all the dependencies pre-downloaded.
            uv_command.append("--offline")

        if _is_list_of(job_commands, str):
            command_args = uv_command + job_commands
            job = executor.submit(submitit.helpers.CommandFunction(command_args))
            logger.info(f"Submitted job {job.job_id}: {command_args}")
            assert isinstance(job, SlurmJob | RemoteSlurmJob)
            return job

        job_commands = typing.cast(list[list[str]], job_commands)
        commands = [uv_command + job_command for job_command in job_commands]
        jobs = executor.submit_array(
            [submitit.helpers.CommandFunction(command) for command in commands]
        )
        assert _is_list_of(jobs, SlurmJob) or _is_list_of(jobs, RemoteSlurmJob)
        for i, (job, command) in enumerate(zip(jobs, commands)):
            logger.info(f"Submitted job {jobs[i].job_id}: {command}")
        return jobs


T = TypeVar("T")


def _is_list_of(v, t: type[T]) -> TypeGuard[list[T]]:
    return isinstance(v, list) and all(isinstance(x, t) for x in v)


@dataclasses.dataclass
class Args:
    hydra: HydraArgs
    resources: SbatchArgs

    executor: SlurmExecutorArgs | RemoteSlurmExecutorArgs = simple_parsing.subgroups(
        # IDEA: Hack simple-parsing to enable dynamic subgroups. Either that, or switch to Hydra
        # itself for this script as well.
        collections.defaultdict(
            lambda: RemoteSlurmExecutorArgs,
            **{"slurm": SlurmExecutorArgs, "remote": RemoteSlurmExecutorArgs},
        ),
        default="slurm",
    )
    """Which cluster to run the job on.

    Use 'current' if you are already connected to a SLURM cluster and want to run your jobs there.
    """

    cluster: str = "current"

    verbose: int = simple_parsing.field(default=0, action="count")
    """Enable verbose output."""


def _setup_logging(verbose: int):
    logging.basicConfig(
        level=logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        # force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )


def main():
    args = simple_parsing.parse(
        Args,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
        # args=argv,
    )

    _setup_logging(args.verbose)

    args_for_each_job = split_arguments_for_each_run(args.hydra.overrides)

    test_job = launch(
        ["pytest", "-x", "-v", "--gen-missing"],
        executor_args=args.executor,
        resources=args.resources,
        cluster=args.cluster,
    )

    sweep_jobs = launch_array(
        [["python", "project/main.py"] + job_args for job_args in args_for_each_job],
        executor_args=args.executor,
        resources=args.resources,
        cluster=args.cluster,
        dependency=Dependency(test_job, type="afterok"),
    )

    for i, (job, job_args) in enumerate(zip(sweep_jobs, args_for_each_job)):
        logger.info(f"Job #{i} ({job.job_id}): {job_args}")

    try:
        logger.debug("Test job results:\n", test_job.result())
    except submitit.core.utils.UncompletedJobError as err:
        logger.error("[bold red blink]Test job failed! Aborting Sweep![/]", extra={"markup": True})
        logger.error(err)
        return

    try:
        job_results = [job.result() for job in sweep_jobs]
        for i, result in enumerate(job_results):
            print(f"Job {i} result: {result}")
    except submitit.core.utils.UncompletedJobError as err:
        logger.error(f"Uncompleted jobs: {err}")
        logger.error(test_job.stderr())
    ## TODO: Launch the final job that uses the best hparams from the sweep and no val split.


def run_with_command_line_args(main_fn: Callable, command_line_args: list[str]):
    @functools.wraps(main_fn)
    def wrapped(*args, **kwargs):
        backup = sys.argv.copy()
        sys.argv[1:] = command_line_args
        result = main_fn()
        sys.argv = backup
        return result

    return wrapped


if __name__ == "__main__":
    main()

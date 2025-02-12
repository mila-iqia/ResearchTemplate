# IDEA: Replacement for the --multirun option of Hydra.
# from hydra.main import   # noqa

import argparse
import contextlib
import dataclasses
import datetime
import functools
import inspect
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import hydra_zen
import rich.logging
import simple_parsing
import submitit
import submitit.core.utils
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra._internal.hydra import Hydra
from hydra._internal.utils import _get_completion_help, create_automatic_config_search_path
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.types import RunMode
from omegaconf import OmegaConf
from submitit.slurm.slurm import _make_sbatch_string

from project.conftest import command_line_overrides  # noqa

logger = logging.getLogger(__name__)


def split_arguments_for_each_run(multirun_overrides: list[str]) -> list[list[str]]:
    overrides_objects = OverridesParser.create().parse_overrides(multirun_overrides)
    batches = BasicSweeper.split_arguments(overrides_objects, max_batch_size=None)
    assert len(batches) == 1  # all jobs in a single batch, since `max_batch_size=None`.
    return batches[0]


def parse_args_into_config(
    args: list[str],
    task_function: Callable,
):
    """Super hacky way to get Hydra to just give us the config for a given set of command-line
    arguments."""
    overrides = args

    # Retrieve the args passed to the `hydra.main` decorator around the task function.
    closure_vars = inspect.getclosurevars(task_function)
    config_name = closure_vars.nonlocals["config_name"]
    config_path = closure_vars.nonlocals["config_path"]
    actual_task_function = closure_vars.nonlocals["task_function"]
    calling_file = inspect.getfile(actual_task_function)

    config_search_path = create_automatic_config_search_path(
        calling_file=calling_file, calling_module=None, config_path=config_path
    )
    hyd = Hydra.create_main_hydra2(task_name="dummy", config_search_path=config_search_path)
    config = hyd.compose_config(
        config_name=config_name,
        overrides=overrides,
        with_log_configuration=True,
        run_mode=RunMode.RUN,
    )
    # This should be the same as `args` I think.
    task_overrides = OmegaConf.to_container(config.hydra.overrides.task, resolve=False)
    assert task_overrides == args
    return config, hyd


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
    cpus_per_task: int = 1
    gpus_per_task: int | str | None = 1
    # ntasks_per_gpu: int | None = None  # todo: support this!
    mem: str = "4G"


@hydra_zen.hydrated_dataclass(submitit.SlurmExecutor)
class SlurmExecutorArgs:
    folder: Path = dataclasses.field(
        default=SCRATCH / "logs" / datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    )
    max_num_timeout: int = 0
    python: str = "uv run --all-extras python"


def launch():
    args = None
    # argv = sys.argv[1:]
    # parser = get_hydra_args_parser()
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH, add_help=False
    )
    parser.add_arguments(HydraArgs, "hydra")

    parser.add_argument(
        "--cluster",
        type=str,
        default="current",
        help=(
            "Which cluster to run the job on. Use 'current' if you are already connected to a "
            "SLURM cluster and want to run your jobs there."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Enable verbose output."
    )
    parser.add_arguments(SlurmExecutorArgs, "executor")
    parser.add_arguments(SbatchArgs, "resources")

    args = parser.parse_args()

    verbose = args.verbose

    overrides: list[str] = args.hydra.overrides
    # todo: use this with the RemoteSlurmExecutor to maybe run the jobs on a remote cluster.
    # todo: maybe use a subgroup action to either parse the remote slurm executor args or regular Executor args.
    # from remote_slurm_executor import RemoteSlurmExecutor  # noqa
    cluster: str = args.cluster
    resources: SbatchArgs = args.resources
    executor_args: SlurmExecutorArgs = args.executor

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

    args_for_each_job = split_arguments_for_each_run(overrides)

    sweep_jobs = []
    # snapshot_dir =
    with (
        submitit.helpers.RsyncSnapshot(snapshot_dir=executor_args.folder, root_dir=None)
        if cluster == "current"
        else contextlib.nullcontext()
    ):
        executor: submitit.SlurmExecutor = hydra_zen.instantiate(executor_args)
        executor.update_parameters(
            **{k: v for k, v in dataclasses.asdict(resources).items() if k != "_target_"}
        )

        ## Launch the test job
        logging.info(f"Working directory: {executor.folder}")

        # idea: Could run tests specific to that particular config (that use some of the job_args?)
        test_command = ["uv", "run", "pytest", "-x", "-v", "--gen-missing"]
        test_job = executor.submit(submitit.helpers.CommandFunction(test_command))
        logger.info(f"Test job ({test_job.job_id}): {test_command}")

        ## Launch the Sweep

        # Kind-of weird that we have to change the executor in-place here.
        additional_parameters = (executor.parameters.get("additional_parameters") or {}).copy()
        additional_parameters["kill-on-invalid-dep"] = "yes"
        executor.update_parameters(
            **{
                "dependency": f"afterok:{test_job.job_id}",
                "additional_parameters": additional_parameters,
            }
        )

        # todo: look into executor._submit_command.
        sweep_jobs = executor.submit_array(
            [
                submitit.helpers.CommandFunction(
                    ["uv", "run", "python", "project/main.py", *job_args]
                )
                for job_args in args_for_each_job
            ]
        )
        for i, (job, job_args) in enumerate(zip(sweep_jobs, args_for_each_job)):
            logger.info(f"Job #{i} ({job.job_id}): {job_args}")

        try:
            logger.debug("Test job results:\n", test_job.result())
        except submitit.core.utils.UncompletedJobError as err:
            logger.error(
                "[bold red blink]Test job failed! Aborting Sweep![/]", extra={"markup": True}
            )
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
    launch()

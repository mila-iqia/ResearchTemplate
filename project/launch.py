# IDEA: Replacement for the --multirun option of Hydra.
# from hydra.main import   # noqa

import contextlib
import dataclasses
import datetime
import functools
import inspect
import logging
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

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


def get_hydra_args_parser() -> simple_parsing.ArgumentParser:
    """Copied from `get_args_parser` in Hydra.

    Uses simple-parsing instead so we can add dataclass args.

    TODO: Make a function that takes an argparse.ArgumentParser and creates a new
    simple_parsing.ArgumentParser by copying its actions.
    """
    from hydra import __version__

    parser = simple_parsing.ArgumentParser(add_help=False, description="Hydra")
    parser.add_argument("--help", "-h", action="store_true", help="Application's help")
    parser.add_argument("--hydra-help", action="store_true", help="Hydra's help")
    parser.add_argument(
        "--version",
        action="version",
        help="Show Hydra's version and exit",
        version=f"Hydra {__version__}",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser.add_argument(
        "--cfg",
        "-c",
        choices=["job", "hydra", "all"],
        help="Show config instead of running [job|hydra|all]",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Used in conjunction with --cfg, resolve config interpolations before printing.",
    )

    parser.add_argument("--package", "-p", help="Config package to show")

    parser.add_argument("--run", "-r", action="store_true", help="Run a job")

    parser.add_argument(
        "--multirun",
        "-m",
        action="store_true",
        help="Run multiple jobs with the configured launcher and sweeper",
    )

    # defer building the completion help string until we actually need to render it
    class LazyCompletionHelp:
        def __repr__(self) -> str:
            return f"Install or Uninstall shell completion:\n{_get_completion_help()}"

    parser.add_argument(
        "--shell-completion",
        "-sc",
        action="store_true",
        help=LazyCompletionHelp(),  # type: ignore
    )

    parser.add_argument(
        "--config-path",
        "-cp",
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        help="Adds an additional config dir to the config search path",
    )

    parser.add_argument(
        "--experimental-rerun",
        help="Rerun a job from a previous config pickle",
    )

    info_choices = [
        "all",
        "config",
        "defaults",
        "defaults-tree",
        "plugins",
        "searchpath",
    ]
    parser.add_argument(
        "--info",
        "-i",
        const="all",
        nargs="?",
        action="store",
        choices=info_choices,
        help=f"Print Hydra information [{'|'.join(info_choices)}]",
    )
    return parser


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
    parser = get_hydra_args_parser()
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

    overrides: list[str] = args.overrides
    # todo: use this with the RemoteSlurmExecutor to maybe run the jobs on a remote cluster.
    # todo: maybe use a subgroup action to either parse the remote slurm executor args or regular Executor args.
    # from remote_slurm_executor import RemoteSlurmExecutor  # noqa
    _cluster: str = args.cluster
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
        # submitit.helpers.RsyncSnapshot(snapshot_dir=executor_args.folder, root_dir=None)
        # if cluster == "current"
        # else SomeCustomRemoteSnapshot?
        contextlib.nullcontext()
    ):
        executor: submitit.SlurmExecutor = hydra_zen.instantiate(executor_args)
        executor.update_parameters(
            **{k: v for k, v in dataclasses.asdict(resources).items() if k != "_target_"}
        )

        ## Launch the test job

        # idea: Could run tests specific to that particular config (that use some of the job_args?)
        test_command = ["uv", "run", "pytest", "-x", "-v", "--gen-missing"]
        logger.info(f"Launching test job with command: {shlex.join(test_command)}")
        test_job = executor.submit(submitit.helpers.CommandFunction(test_command))

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

        for i, job_args in enumerate(args_for_each_job):
            logger.info(f"Job {i}: {job_args}")
        # todo: look into executor._submit_command.
        sweep_jobs = executor.submit_array(
            [
                submitit.helpers.CommandFunction(
                    ["uv", "run", "python", "project/main.py", *job_args]
                )
                for i, job_args in enumerate(args_for_each_job)
            ]
        )

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

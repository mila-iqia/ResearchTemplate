# IDEA: Replacement for the --multirun option of Hydra.
# from hydra.main import   # noqa

import inspect
import os
from collections.abc import Callable
from pathlib import Path

import submitit
from omegaconf import OmegaConf


def split_arguments_for_each_run(argv: list[str] | None) -> list[list[str]]:
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra._internal.utils import get_args_parser
    from hydra.core.override_parser.overrides_parser import OverridesParser

    cli_overrides = get_args_parser().parse_args(argv).overrides
    overrides_objects = OverridesParser.create().parse_overrides(cli_overrides)
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

    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path
    from hydra.types import RunMode

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


def launch():
    args = None
    # argv = sys.argv[1:]

    args_for_each_job = split_arguments_for_each_run(args)

    jobs = []
    snapshot_dir = SCRATCH / "foo"
    with submitit.helpers.RsyncSnapshot(snapshot_dir=snapshot_dir, root_dir=None):
        executor = submitit.SlurmExecutor(snapshot_dir, max_num_timeout=0)
        executor.update_parameters(
            time=10,
            gpus_per_task=1,
            cpus_per_task=1,
            mem="4G",
        )
        jobs = executor.submit_array(
            [
                submitit.helpers.CommandFunction(
                    ["uv", "run", "python", "project/main.py", *job_args]
                )
                for i, job_args in enumerate(args_for_each_job)
            ]
        )
    # for i, job_args in enumerate(args_for_each_job):
    #     print(f"Job {i}: {job_args}")

    print([job.result() for job in jobs])


if __name__ == "__main__":
    launch()

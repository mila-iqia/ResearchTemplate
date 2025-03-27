# IDEA: Replacement for the --multirun option of Hydra.
# from hydra.main import   # noqa

import copy
import datetime as dt
import logging
import os
import pathlib
from pathlib import Path

import xm_slurm
import xm_slurm.contrib.clusters
from absl import app
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from xmanager import xm

# TODO: Look into Fiddle

# note: port is from the *local* SSH config entry for mila / compute nodes
os.environ["DOCKER_HOST"] = "tcp://127.0.0.1:2366"

logger = logging.getLogger(__name__)
SCRATCH = Path(os.environ["SCRATCH"])


def split_arguments_for_each_run(multirun_overrides: list[str]) -> list[list[str]]:
    overrides_objects = OverridesParser.create().parse_overrides(multirun_overrides)
    batches = BasicSweeper.split_arguments(overrides_objects, max_batch_size=None)
    assert len(batches) == 1  # all jobs in a single batch, since `max_batch_size=None`.
    return batches[0]


@xm.run_in_asyncio_loop
async def main(_):
    async with xm_slurm.create_experiment("My Experiment") as experiment:
        # Step 1: Specify the executor specification
        executor_spec = xm_slurm.Slurm.Spec(tag="ghcr.io/lebrice/xm-slurm/test:latest")

        # Step 2: Specify the executable and package it
        [test_executable, train_executable] = experiment.package(
            [
                xm_slurm.uv_container(
                    executor_spec=executor_spec,
                    # todo: add a way to pass the --all-extras arg to `uv`.
                    entrypoint=xm.ModuleName("pytest"),
                ),
                xm_slurm.uv_container(
                    executor_spec=executor_spec,
                    # note: args is passed to the entrypoint (e.g. job args)
                    # todo: add a way to pass the --all-extras arg to `uv`.
                    entrypoint=xm.CommandList(["project/main.py"]),
                ),
            ],
        )

        workdir = pathlib.Path(f"$SCRATCH/xm-slurm-examples/{experiment.experiment_id}")

        # Step 3: Schedule test job
        executor = xm_slurm.Slurm(
            requirements=xm_slurm.JobRequirements(
                CPU=1,
                RAM=1.0 * xm.GiB,
                GPU=1,
                replicas=1,
                cluster=xm_slurm.contrib.clusters.mila(),
            ),
            time=dt.timedelta(hours=1),
        )
        train_executor = copy.deepcopy(executor)
        test_executor = copy.deepcopy(executor)

        test_job = xm.Job(
            executable=test_executable,
            executor=test_executor,
            args=["-x", "-v", "--gen-missing"],
        )
        train_job = xm.Job(
            executable=train_executable,
            executor=train_executor,
            args=["experiment=example", "trainer.fast_dev_run=True", f"+hydra.run.dir={workdir}"],
        )

        test_job_work_unit = await experiment.add(test_job)
        train_job_work_unit = await experiment.add(
            train_job, dependency=test_job_work_unit.after_completed()
        )

        await train_job_work_unit.wait_until_complete()
        print(
            f"Work Unit {train_job_work_unit!r} finished executing with status {await train_job_work_unit.get_status()}"
        )


if __name__ == "__main__":
    app.run(main)

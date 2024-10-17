# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/hydra/blob/main/examples/plugins/example_launcher_plugin/hydra_plugins/example_launcher_plugin/example_launcher.py

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict
from remote_slurm_executor.slurm_remote import RemoteSlurmExecutor

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    BaseSubmititLauncher,
    SlurmLauncher,  # noqa
)


@dataclass
class RemoteSlurmQueueConf(SlurmQueueConf):
    """Slurm configuration overrides and specific parameters."""

    _target_: str = (
        "hydra_plugins.remote_slurm_launcher._remote_launcher_plugin.RemoteSlurmLauncher"
    )
    cluster_hostname: str = "mila"
    repo_dir_on_cluster: str | None = None


class RemoteSlurmLauncher(BaseSubmititLauncher):
    _EXECUTOR = "remoteslurm"

    def __init__(self, **params) -> None:
        # self.cluster = cluster
        super().__init__(**params)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        executor = RemoteSlurmExecutor(
            folder=self.params["submitit_folder"],
            cluster_hostname=self.params["cluster_hostname"],
            repo_dir_on_cluster=self.params.get("repo_dir_on_cluster"),
        )
        # specify resources/parameters
        executor.update_parameters(
            **{
                x: y
                for x, y in params.items()
                if x not in ["submitit_folder", "cluster_hostname", "repo_dir_on_cluster"]
            }
        )

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : " f"{self.config.hydra.sweep.dir}"
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

        jobs = executor.map_array(self, *zip(*job_params))
        return [j.results()[0] for j in jobs]


# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)


class ExampleLauncher(Launcher):
    def __init__(self, foo: str, bar: str) -> None:
        self.config: DictConfig | None = None
        self.task_function: TaskFunction | None = None
        self.hydra_context: HydraContext | None = None

        # foo and var are coming from the the plugin's configuration
        self.foo = foo
        self.bar = bar

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """
        :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
        :param initial_job_idx: Initial job idx in batch.
        :return: an array of return values from run_job with indexes corresponding to the input list indexes.
        """
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Example Launcher(foo={self.foo}, bar={self.bar}) is launching {len(job_overrides)} jobs locally"
        )
        log.info(f"Sweep output dir : {sweep_dir}")
        runs = []

        # Initialize custom executor
        executor = RemoteSlurmExecutor(self.foo, self.bar)

        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            with open_dict(sweep_config):
                # This typically coming from the underlying scheduler (SLURM_JOB_ID for instance)
                # In that case, it will not be available here because we are still in the main process.
                # but instead should be populated remotely before calling the task_function.
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx

            ret = executor.submit()
            runs.append(ret)
            # If your launcher is executing code in a different process, it is important to restore
            # the singleton state in the new process.
            # To do this, you will likely need to serialize the singleton state along with the other
            # parameters passed to the child process.

            # happening on this process (executing launcher)
            state = Singleton.get_state()

            # happening on the spawned process (executing task_function in run_job)
            Singleton.set_state(state)

            # ret = run_job(
            #    hydra_context=self.hydra_context,
            #    task_function=self.task_function,
            #    config=sweep_config,
            #    job_dir_key="hydra.sweep.dir",
            #    job_subdir_key="hydra.sweep.subdir",
            # )
            # runs.append(ret)
            # reconfigure the logging subsystem for Hydra as the run_job call configured it for the Job.
            # This is needed for launchers that calls run_job in the same process and not spawn a new one.
            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        return runs

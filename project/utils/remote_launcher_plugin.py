# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/hydra/blob/main/examples/plugins/example_launcher_plugin/hydra_plugins/example_launcher_plugin/example_launcher.py

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra.plugins.plugin import Plugin
from hydra.utils import instantiate
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from omegaconf import DictConfig
from remote_slurm_executor.slurm_remote import RemoteSlurmExecutor

logger = logging.getLogger(__name__)


def _instantiate(self: Plugins, config: DictConfig) -> Plugin:
    """FIX annoying Hydra thing: plugins have to be in an "approved" `hydra_plugins` namespace?!"""
    from hydra._internal import utils as internal_utils

    classname = internal_utils._get_cls_name(config, pop=False)
    try:
        if classname is None:
            raise ImportError("class not configured")

        if not self.is_in_toplevel_plugins_module(classname):
            # All plugins must be defined inside the approved top level modules.
            # For plugins outside of hydra-core, the approved module is hydra_plugins.
            if classname == RemoteSlurmQueueConf._target_:
                return instantiate(config)
            # raise RuntimeError(f"Invalid plugin '{classname}': not the hydra_plugins package")

        if classname not in self.class_name_to_class.keys():
            raise RuntimeError(f"Unknown plugin class : '{classname}'")
        clazz = self.class_name_to_class[classname]
        plugin = instantiate(config=config, _target_=clazz)
        assert isinstance(plugin, Plugin)

    except ImportError as e:
        raise ImportError(
            f"Could not instantiate plugin {classname} : {str(e)}\n\n\tIS THE PLUGIN INSTALLED?\n\n"
        )

    return plugin


Plugins._instantiate = _instantiate


@dataclass
class RemoteSlurmQueueConf(SlurmQueueConf):
    """Slurm configuration overrides and specific parameters."""

    _target_: str = "project.utils.remote_launcher_plugin.RemoteSlurmLauncher"

    cluster_hostname: str = "mila"
    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"
    internet_access_on_compute_nodes: bool = False


class RemoteSlurmLauncher(BaseSubmititLauncher):
    _EXECUTOR = "remoteslurm"

    def __init__(
        self,
        cluster_hostname: str,
        submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j",
        internet_access_on_compute_nodes: bool | None = None,
        # maximum time for the job in minutes
        timeout_min: int = 60,
        # number of cpus to use for each task
        cpus_per_task: int | None = None,
        # number of gpus to use on each node
        gpus_per_node: int | None = None,
        # number of tasks to spawn on each node
        tasks_per_node: int = 1,
        # memory to reserve for the job on each node (in GB)
        mem_gb: int | None = None,
        # number of nodes to use for the job
        nodes: int = 1,
        # name of the job
        name: str = "${hydra.job.name}",
        # redirect stderr to stdout
        stderr_to_stdout: bool = False,
        partition: str | None = None,
        qos: str | None = None,
        comment: str | None = None,
        constraint: str | None = None,
        exclude: str | None = None,
        gres: str | None = None,
        cpus_per_gpu: int | None = None,
        gpus_per_task: int | None = None,
        mem_per_gpu: str | None = None,
        mem_per_cpu: str | None = None,
        account: str | None = None,
        signal_delay_s: int = 120,
        max_num_timeout: int = 0,
        additional_parameters: dict[str, Any] | None = None,
        array_parallelism: int = 256,
        setup: list[str] | None = None,
    ) -> None:
        # self.cluster = cluster
        super().__init__(
            cluster_hostname=cluster_hostname,
            submitit_folder=submitit_folder,
            internet_access_on_compute_nodes=internet_access_on_compute_nodes,
            timeout_min=timeout_min,
            cpus_per_task=cpus_per_task,
            gpus_per_node=gpus_per_node,
            tasks_per_node=tasks_per_node,
            mem_gb=mem_gb,
            nodes=nodes,
            name=name,
            stderr_to_stdout=stderr_to_stdout,
            partition=partition,
            qos=qos,
            comment=comment,
            constraint=constraint,
            exclude=exclude,
            gres=gres,
            cpus_per_gpu=cpus_per_gpu,
            gpus_per_task=gpus_per_task,
            mem_per_gpu=mem_per_gpu,
            mem_per_cpu=mem_per_cpu,
            account=account,
            signal_delay_s=signal_delay_s,
            max_num_timeout=max_num_timeout,
            additional_parameters=additional_parameters,
            array_parallelism=array_parallelism,
            setup=setup,
        )

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
            internet_access_on_compute_nodes=self.params["internet_access_on_compute_nodes"],
            # repo_dir_on_cluster=self.params.get("repo_dir_on_cluster"),
        )
        # Do *not* overwrite the `setup` if it's already in the executor's parameters!
        if _setup := params.get("setup"):
            executor.parameters["setup"] = (executor.parameters.get("setup", []) or []) + _setup

        executor.update_parameters(
            **{
                x: y
                for x, y in params.items()
                if x
                not in [
                    "submitit_folder",
                    "cluster_hostname",
                    "max_num_timeout",
                    "mem_gb",
                    "name",
                    "tasks_per_node",
                    "timeout_min",
                    "setup",
                ]
            }
        )

        logger.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : " f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: list[tuple[list[str], str, int, str, dict]] = []
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

        jobs = executor.map_array(self, *zip(*job_params))
        return [j.results()[0] for j in jobs]

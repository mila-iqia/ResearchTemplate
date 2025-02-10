from __future__ import annotations

import logging
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hydra.conf import HydraConf
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn

from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf, SlurmQueueConf  # noqa
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher

if typing.TYPE_CHECKING:
    from submitit import Job


log = logging.getLogger(__name__)


@runtime_checkable
class TopLevelConfig(Protocol):
    hydra: HydraConf


class PytestJobDependencyLauncher(SlurmLauncher):
    """IDEA: launcher that uses a test job before a sweep and makes the sweep jobs depend on it."""

    def __init__(
        self,
        folder: str | Path,
        test_command: Sequence[str] = ("pytest", "-x", "-v"),
        job_name: str = "${hydra.job.name}",
        partition: str | None = None,
        time: int = 5,
        nodes: int = 1,
        ntasks_per_node: int | None = None,
        cpus_per_task: int | None = None,
        cpus_per_gpu: int | None = None,
        num_gpus: int | None = None,  # legacy
        gpus_per_node: int | None = None,
        gpus_per_task: int | None = None,
        qos: str | None = None,  # quality of service
        setup: list[str] | None = None,
        mem: str | None = None,
        mem_per_gpu: str | None = None,
        mem_per_cpu: str | None = None,
        signal_delay_s: int = 90,
        comment: str | None = None,
        constraint: str | None = None,
        exclude: str | None = None,
        account: str | None = None,
        gres: str | None = None,
        mail_type: str | None = None,
        mail_user: str | None = None,
        nodelist: str | None = None,
        dependency: str | None = None,
        exclusive: bool | str | None = None,
        array_parallelism: int = 256,
        wckey: str = "submitit",
        stderr_to_stdout: bool = False,
        additional_parameters: dict[str, Any] | None = None,
        srun_args: Iterable[str] | None = None,
        use_srun: bool = True,
    ) -> None:
        super().__init__(
            folder=folder,
            job_name=job_name,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
            cpus_per_gpu=cpus_per_gpu,
            num_gpus=num_gpus,
            gpus_per_node=gpus_per_node,
            gpus_per_task=gpus_per_task,
            qos=qos,
            setup=setup,
            mem=mem,
            mem_per_gpu=mem_per_gpu,
            mem_per_cpu=mem_per_cpu,
            signal_delay_s=signal_delay_s,
            comment=comment,
            constraint=constraint,
            exclude=exclude,
            account=account,
            gres=gres,
            mail_type=mail_type,
            mail_user=mail_user,
            nodelist=nodelist,
            dependency=dependency,
            exclusive=exclusive,
            array_parallelism=array_parallelism,
            wckey=wckey,
            stderr_to_stdout=stderr_to_stdout,
            additional_parameters=additional_parameters,
            srun_args=srun_args,
            use_srun=use_srun,
        )
        self.test_job_params = test_command
        assert self.config is None or isinstance(self.config, TopLevelConfig)
        self.test_job: Job[str] | None = None

    def __call__(
        self,
        sweep_overrides: list[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: dict[type, Singleton],
    ) -> JobReturn:
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None
        # IDEA: Could add a suffix to make a job into a "debugging" job, like
        # "+trainer.fast_dev_run=True" or "+trainer.max_epochs=1"
        # if sweep_overrides == list(self.test_job_params):
        #     # test job. Run pytest.

        return super().__call__(
            sweep_overrides=sweep_overrides,
            job_dir_key=job_dir_key,
            job_num=job_num,
            job_id=job_id,
            singleton_state=singleton_state,
        )

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit
        from submitit.core.utils import CommandFunction

        if initial_job_idx == 0:
            assert self.test_job is None
            # Create the test job and save it on `self`.
            executor = submitit.SlurmExecutor(
                folder=self.params["folder"],
                max_num_timeout=self.params.get("max_num_timeout", 3),
            )
            params = {
                k: v
                for k, v in self.params.items()
                if k not in ["max_num_timeout", "submitit_folder"]
            }
            executor.update_parameters(**params)
            self.test_job = executor.submit(CommandFunction(list(self.test_job_params)))
            print(f"Launched test job: {self.test_job.job_id} for command {self.test_job_params}")

            # Unfortunately have to modify the `self.params` dict in-place, because the `SlurmLauncher`
            # uses it to create the executor and then launch the job.
            additional_parameters_ = self.params.setdefault("additional_parameters", {})
            if "dependency" in additional_parameters_:
                raise RuntimeError("Jobs can't already have a dependency!")

            additional_parameters_["dependency"] = f"afterok:{self.test_job.job_id}"

        # We already launched the test job.
        assert self.test_job is not None
        assert "additional_parameters" in self.params
        assert "dependency" in self.params["additional_parameters"]
        assert (
            self.params["additional_parameters"]["dependency"] == f"afterok:{self.test_job.job_id}"
        )
        return super().launch(job_overrides, initial_job_idx)

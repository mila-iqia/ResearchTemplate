"""Idea: Use `submitit` to test that the setup works for this repo on the current cluster.

TODOs/ideas:
- Create a fixture that scancel's the jobs if I KeyboardInterrupt the test.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import warnings
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Literal, overload

import pytest
from hydra import initialize
from hydra._internal.callbacks import Callbacks
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobReturn, JobStatus, run_job
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, RunMode
from hydra_zen._launch import _NotSet, _store_config
from omegaconf import OmegaConf

from project.configs.config import Config
from project.main import main
from project.utils.hydra_utils import instantiate
from project.utils.types import Dataclass

PROJECT_NAME = "project"  # TODO: Change this to the name of your project.
TEST_JOB_NAME = "cluster_test"

# TODO: get output of savail instead.
gpu_types = [
    "1g.10gb",  # MIG-ed A100 GPU
    "2g.20gb",  # MIG-ed A100 GPU
    "3g.40gb",  # MIG-ed A100 GPU
    # "a100",
    # "a100l",  # Note: needs a reservation.
    # "a6000",
    "rtx8000",
    pytest.param(
        "v100",
        marks=[
            pytest.mark.xfail(reason="Can take a while to schedule"),
            pytest.mark.timeout(120),
        ],
    ),
]
gpu_types = [None]

pytestmark = pytest.mark.skipif(
    not shutil.which("sbatch"), reason="Needs to be run on a SLURM cluster."
)


@pytest.fixture(autouse=True, scope="session")
def scancel_jobs_after_tests():
    yield
    # TODO: run scancel over ssh to the cluster if running locally.
    if shutil.which("scancel"):
        username = os.environ["USER"]
        subprocess.check_call(["scancel", "-u", username, "--name", TEST_JOB_NAME])


@pytest.mark.skipif("SLURM_JOB_ID" not in os.environ, reason="Not running on a SLURM cluster")
@pytest.mark.parametrize("nodes", [1])
@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("gpu_type", gpu_types)
def test_launch_job_on_cluster(nodes: int, gpus: int, gpu_type: str | None) -> None:
    """Test that we can auto-pack multiple jobs on a single GPU."""

    jobs_per_gpu = 2

    cpus_per_gpu = 2
    mem_per_node_gb = 16
    assert gpus % nodes == 0
    gpus_per_node = gpus // nodes
    mem_per_gpu_gb = max(1, math.floor(mem_per_node_gb / gpus_per_node))

    if gpus > 1 and gpu_type and "." in gpu_type:
        # NOTE: Is it possible that this would actually work?
        pytest.skip("Not launching multi-gpu jobs using MIG.")

    overrides = [
        "name=test",
        "algorithm=example_algo",
        "datamodule=cifar10",
        "experiment=overfit_one_batch",
        "trainer.max_epochs=1",
        # This 'resources' group is where most of the configuration options for the slurm
        # launcher are. Here we just overwrite some of them.
        # For more info, check out project/configs/resources/one_gpu.yaml
        "resources=one_gpu",
        # Overrides compared to `one_gpu.yaml`:
        # TODO: Pack more than one job on a single GPU.
        # Jobs should last less than 10 minutes.  (actually more like 1 minute, but making it
        # simple.)
        "hydra.launcher.additional_parameters.time=0-00:10:00",
        f"hydra.launcher.nodes={nodes}",
        f"hydra.launcher.tasks_per_node={gpus_per_node * jobs_per_gpu}",  # a.k.a. ntasks_per_node
        f"hydra.launcher.cpus_per_task={min(1, cpus_per_gpu // jobs_per_gpu)}",
        "hydra.launcher.gres=" + (f"gpu:{gpu_type}:1" if gpu_type is not None else "gpu:1"),
        f"hydra.launcher.gres=gpu:{gpu_type}:1",
        f"hydra.launcher.mem_per_gpu={mem_per_gpu_gb}G",
        # f"hydra.launcher.mem_gb={mem_gb}",
        f"trainer.devices={gpus}",
    ]

    distributed_training = gpus > 1 or nodes > 1

    if gpu_type == "" and distributed_training:
        # Avoid the nodes with MIG-ed GPUs when asking for "any" GPU in a distributed setting.
        overrides.append("hydra.launcher.additional_parameters.exclude=cn-g[005-012,017-026]")
    if distributed_training:
        overrides.append("+trainer.strategy=ddp")
    if nodes > 1:
        overrides.append(f"trainer.num_nodes={nodes}")  # TODO: Actually test nodes > 1

    # Run the job directly on the current node:
    # output = main(config)
    config_path = "configs"
    output = launch(
        Config,
        task_function=main,
        overrides=overrides,
        multirun=True,
        config_name="config",
        job_name=TEST_JOB_NAME,
        config_path=config_path,
        caller_stack_depth=2,
    )
    job_outputs = output
    assert len(job_outputs) == 1
    assert len(job_outputs[0]) == 1
    job_output = job_outputs[0][0]
    assert job_output.status is JobStatus.COMPLETED
    job_val_classification_error = job_output.return_value
    assert isinstance(job_val_classification_error, float)
    assert 0 <= job_val_classification_error <= 1
    # options = OmegaConf.to_object(config)


def test_packing_runs_in_one_job() -> None:
    """Test that we can pack multiple runs in a single job (on one GPU)."""
    config_path = "configs"

    nodes = 1
    gpus = 1
    # gpu_type = "1g.10gb"
    gpu_type = None
    cpus_per_gpu = 2
    mem_per_node_gb = 16
    assert gpus % nodes == 0
    gpus_per_node = gpus // nodes
    mem_per_gpu_gb = max(1, math.floor(mem_per_node_gb / gpus_per_node))

    if gpus > 1 and gpu_type and "." in gpu_type:
        # NOTE: Is it possible that this would actually work?
        pytest.skip("Not launching multi-gpu jobs using MIG.")

    overrides = [
        "name=test",
        "algorithm=example_algo",
        "datamodule=cifar10",
        "experiment=overfit_one_batch",
        "trainer.max_epochs=1",
        # This 'resources' group is where most of the configuration options for the slurm
        # launcher are. Here we just overwrite some of them.
        # For more info, check out project/configs/resources/one_gpu.yaml
        "resources=one_gpu",
        # Overrides compared to `one_gpu.yaml`:
        # TODO: Pack more than one job on a single GPU.
        # Jobs should last less than 10 minutes.  (actually more like 1 minute, but making it
        # simple.)
        "hydra.launcher.additional_parameters.time=0-00:10:00",
        f"hydra.launcher.nodes={nodes}",
        f"hydra.launcher.tasks_per_node={gpus_per_node}",  # a.k.a. ntasks_per_node
        f"hydra.launcher.cpus_per_task={cpus_per_gpu}",
        "hydra.launcher.gpus_per_task=" + (f"{gpu_type}:1" if gpu_type is not None else "1"),
        f"hydra.launcher.mem_per_gpu={mem_per_gpu_gb}G",
        # f"hydra.launcher.mem_gb={mem_gb}",
        f"trainer.devices={gpus}",
    ]

    distributed_training = gpus > 1 or nodes > 1

    if gpu_type == "" and distributed_training:
        # Avoid the nodes with MIG-ed GPUs when asking for "any" GPU in a distributed setting.
        overrides.append("hydra.launcher.additional_parameters.exclude=cn-g[005-012,017-026]")
    if distributed_training:
        overrides.append("+trainer.strategy=ddp")
    if nodes > 1:
        overrides.append(f"trainer.num_nodes={nodes}")  # TODO: Actually test nodes > 1

    # Run the job directly on the current node:
    # output = main(config)

    output = launch(
        Config,
        task_function=main,
        overrides=overrides,
        multirun=True,
        config_name="config",
        job_name=TEST_JOB_NAME,
        config_path=config_path,
        caller_stack_depth=2,
    )
    job_outputs = output
    assert len(job_outputs) == 1
    assert len(job_outputs[0]) == 1
    job_output = job_outputs[0][0]
    assert job_output.status is JobStatus.COMPLETED
    job_val_classification_error = job_output.return_value
    assert isinstance(job_val_classification_error, float)
    assert 0 <= job_val_classification_error <= 1
    # options = OmegaConf.to_object(config)


@overload
def launch(
    config: Dataclass | type[Dataclass] | Mapping[str, Any],
    task_function: Callable[[Any], Any],
    overrides: list[str] | None = None,
    version_base: str | type[_NotSet] | None = _NotSet,
    to_dictconfig: bool = False,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    # This changes:
    multirun: Literal[True] = True,
    # Added parameters:
    config_path: str | None = None,
    caller_stack_depth: int = 2,
) -> list[list[JobReturn]]: ...


@overload
def launch(
    config: Dataclass | type[Dataclass] | Mapping[str, Any],
    task_function: Callable[[Any], Any],
    overrides: list[str] | None = None,
    version_base: str | type[_NotSet] | None = _NotSet,
    to_dictconfig: bool = False,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    # This changes:
    multirun: Literal[False] = False,
    # Added parameters:
    config_path: str | None = None,
    caller_stack_depth: int = 2,
) -> JobReturn: ...


# NOTE: This is a copied and slightly modified version of `launch` from `hydra_zen._launch` to add
# the `config_path` and `caller_stack_depth` parameters.
def launch(
    config: Dataclass | type[Dataclass] | Mapping[str, Any],
    task_function: Callable[[Any], Any],
    overrides: list[str] | None = None,
    version_base: str | type[_NotSet] | None = "1.2",
    to_dictconfig: bool = False,
    config_name: str = "zen_launch",
    job_name: str = "zen_launch",
    with_log_configuration: bool = True,
    multirun: bool = False,
    # Added parameters:
    config_path: str | None = None,
    caller_stack_depth: int = 2,
) -> JobReturn | list[list[JobReturn]]:
    r"""Launch a Hydra job using a Python-based interface.

    `launch` is designed to closely match the interface of the standard Hydra CLI.
    For example, launching a Hydra job from the CLI via::

       $ python my_task.py job/group=group_name job.group.param=1

    corresponds to the following usage of `launch`:

       >>> job = launch(  # doctest: +SKIP
       ...     config,
       ...     task_function,
       ...     overrides=["job/group=group_name", "job.group.param=1"],
       ... )

    Parameters
    ----------
    config : Dataclass | Type[Dataclass] | Mapping[str, Any]
        A config that will be passed to ``task_function``.

    task_function : Callable[[DictConfig], Any]
        The function that Hydra will execute. Its input will be ``config``, which
        has been modified via the specified ``overrides``

    overrides : Optional[List[str]]
        If provided, sets/overrides values in ``config``. See [1]_ and [2]_
        for a detailed discussion of the "grammar" supported by ``overrides``.

    multirun : bool (default: False)
        Launch a Hydra multi-run ([3]_).

    version_base : Optional[str], optional (default=_NotSet)
        Available starting with Hydra 1.2.0.
        - If the `version_base parameter` is not specified, Hydra 1.x will use defaults compatible
          with version 1.1. Also in this case, a warning is issued to indicate an explicit
          version_base is preferred.
        - If the `version_base parameter` is `None`, then the defaults are chosen for the current
          minor Hydra version. For example for Hydra 1.2, then would imply `config_path=None` and
          `hydra.job.chdir=False`.
        - If the `version_base` parameter is an explicit version string like "1.1", then the
          defaults appropriate to that version are used.

    to_dictconfig: bool (default: False)
        If ``True``, convert a ``dataclasses.dataclass`` to a ``omegaconf.DictConfig``. Note, this
        will remove Hydra's cabability for validation with structured configurations.

    config_name : str (default: "zen_launch")
        Name of the stored configuration in Hydra's ConfigStore API.

    job_name : str (default: "zen_launch")

    with_log_configuration : bool (default: True)
        If ``True``, enables the configuration of the logging subsystem from the loaded config.

    Returns
    -------
    result : JobReturn | Any
        If ``multirun is False``:
            A ``JobReturn`` object storing the results of the Hydra experiment via the following
                attributes
                - ``cfg``: Reflects ``config``
                - ``overrides``: Reflects ``overrides``
                - ``return_value``: The return value of the task function
                - ``hydra_cfg``: The Hydra configuration object
                - ``working_dir``: The experiment working directory
                - ``task_name``: The task name of the Hydra job
                - ``status``: A ``JobStatus`` enum reporting whether or not the job completed
                  successfully
        Else:
            Return values of all launched jobs (depends on the Sweeper implementation).

    References
    ----------
    .. [1] https://hydra.cc/docs/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/configure_hydra/intro
    .. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run

    Examples
    --------

    **Basic usage**

    Let's define and launch a trivial Hydra app.

    >>> from hydra_zen import make_config, launch, to_yaml  # doctest: +SKIP

    First, we will define a config, which determines the configurable interface to our
    "app". For the purpose of example, we'll design the "interface" of this config to accept
    two configurable parameters: ``a`` and ``b``.

    >>> Conf = make_config("a", "b")  # doctest: +SKIP

    Our task function accepts the config as an input and uses it to run some generic functionality.
    For simplicity's sake, let's design this task function to: convert the job's config to a
    yaml-formatted string, print it, and then return the string.

    >>> def task_fn(cfg):
    ...     out = to_yaml(cfg)  # task's input config, converted to yaml-string
    ...     print(out)
    ...     return out

    Now, let's use `launch` to run this task function via Hydra, using particular configured
    values (or, "overrides") for ``a`` and ``b``.

    >>> job_out = launch(Conf, task_fn, overrides=["a=1", "b='foo'"])  # doctest: +SKIP
    a: 1
    b: foo

    Let's inspect ``job_out`` to see the ways that it summarizes the results of this job.

    >>> job_out.return_value  # the value returned by `task_fn` # doctest: +SKIP
    'a: 1\nb: foo\n'

    >>> # where the job's outputs, logs, and configs are saved
    >>> job_out.working_dir # doctest: +SKIP
    'outputs/2021-10-19/15-27-11'

    >>> job_out.cfg  # the particular config used to run our task-function  # doctest: +SKIP
    {'a': 1, 'b': 'foo'}

    >>> job_out.overrides  # the overrides that we provides  # doctest: +SKIP
    ['a=1', "b='foo'"]

    >>> job_out.status  # the job's completion status  # doctest: +SKIP
    <JobStatus.COMPLETED: 1>

    **Launching a multirun job**

    We can launch multiple runs of our task-function, using various configured values.
    Let's launch a multirun that sweeps over three configurations

    >>> (outputs,) = launch(  # doctest: +SKIP
    ...     Conf,
    ...     task_fn,
    ...     overrides=["a=1,2,3", "b='bar'"],
    ...     multirun=True,
    ... )
    [2021-10-19 17:50:07,334][HYDRA] Launching 3 jobs locally
    [2021-10-19 17:50:07,334][HYDRA] 	#0 : a=1 b='bar'
    a: 1
    b: bar
    [2021-10-19 17:50:07,434][HYDRA] 	#1 : a=2 b='bar'
    a: 2
    b: bar
    [2021-10-19 17:50:07,535][HYDRA] 	#2 : a=3 b='bar'
    a: 3
    b: bar

    ``outputs`` contains three corresponding ``JobReturns`` instances.

    >>> len(outputs)  # doctest: +SKIP
    3
    >>> [j.cfg for j in outputs]  # doctest: +SKIP
    [{'a': 1, 'b': 'bar'}, {'a': 2, 'b': 'bar'}, {'a': 3, 'b': 'bar'}]

    Each run's outputs, logs, and configs are saved to separate working directories

    >>> [j.working_dir for j in outputs]  # doctest: +SKIP
    ['multirun/2021-10-19/17-50-07\\0',
    'multirun/2021-10-19/17-50-07\\1',
    'multirun/2021-10-19/17-50-07\\2']
    """

    # used for check below
    _num_dataclass_fields = 0
    if is_dataclass(config):
        _num_dataclass_fields = len(fields(config))

    # store config in ConfigStore
    if to_dictconfig and is_dataclass(config):
        # convert Dataclass to a DictConfig
        dictconfig = OmegaConf.create(OmegaConf.to_container(OmegaConf.structured(config)))
        config_name = _store_config(dictconfig, config_name)
    else:
        config_name = _store_config(config, config_name)

    # Initializes Hydra and add the config_path to the config search path
    with initialize(
        config_path=config_path,
        caller_stack_depth=caller_stack_depth,
        job_name=job_name,
        version_base=version_base,
    ):
        # taken from hydra.compose with support for MULTIRUN
        gh = GlobalHydra.instance()
        assert gh.hydra is not None

        # Load configuration
        cfg = gh.hydra.compose_config(
            config_name=config_name,
            overrides=overrides if overrides is not None else [],
            run_mode=RunMode.RUN if not multirun else RunMode.MULTIRUN,
            from_shell=False,
            with_log_configuration=with_log_configuration,
        )

        callbacks = Callbacks(cfg)
        run_start = callbacks.on_run_start if not multirun else callbacks.on_multirun_start
        run_start(config=cfg, config_name=config_name)

        hydra_context = HydraContext(config_loader=gh.config_loader(), callbacks=callbacks)

        if not multirun:
            job = run_job(
                hydra_context=hydra_context,
                task_function=task_function,
                config=cfg,
                job_dir_key="hydra.run.dir",
                job_subdir_key=None,
                configure_logging=with_log_configuration,
            )
            callbacks.on_run_end(config=cfg, config_name=config_name, job_return=job)

            # access the result to trigger an exception in case the job failed.
            _ = job.return_value
        else:
            # Instantiate sweeper without using Hydra's Plugin discovery (Zen!)
            sweeper = instantiate(cfg.hydra.sweeper)
            assert isinstance(sweeper, Sweeper)
            sweeper.setup(
                config=cfg,
                hydra_context=hydra_context,
                task_function=task_function,
            )

            task_overrides = OmegaConf.to_container(cfg.hydra.overrides.task, resolve=False)
            assert isinstance(task_overrides, list)
            job = sweeper.sweep(arguments=task_overrides)
            callbacks.on_multirun_end(config=cfg, config_name=config_name)

    if is_dataclass(config):
        _num_dataclass_fields_after = len(fields(config))
        if (
            _num_dataclass_fields_after == 0
            and _num_dataclass_fields_after < _num_dataclass_fields
        ):
            warnings.warn(
                "Your dataclass-based config was mutated by this run. If you just executed with a "
                "`hydra/launcher` that utilizes cloudpickle (e.g., hydra-submitit-launcher), "
                "there is a known issue with dataclasses "
                "(see: https://github.com/cloudpipe/cloudpickle/issues/386). You will have "
                "to restart your interactive environment to run `launch` again. To avoid this "
                "issue you can use the `launch` option: `to_dictconfig=True`."
            )

    return job

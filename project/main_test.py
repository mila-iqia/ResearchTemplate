# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shutil
import sys
import uuid
from unittest.mock import Mock

import hydra_zen
import omegaconf.errors
import pytest
import torch
from _pytest.mark.structures import ParameterSet
from hydra.types import RunMode
from omegaconf import DictConfig

import project.main
from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.conftest import command_line_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.testutils import IN_GITHUB_CI

from .main import PROJECT_NAME, main

CONFIG_DIR = REPO_ROOTDIR / PROJECT_NAME / "configs"


def test_jax_can_use_the_GPU():
    """Test that Jax can use the GPU if it we have one."""
    # NOTE: Super interesting: Seems like running just an
    # `import jax.numpy; print(jax.numpy.zeros(1).devices())` in a new terminal FAILS, but if you
    # do `import torch` before that, then it works!
    import jax.numpy

    device = jax.numpy.zeros(1).devices().pop()
    if shutil.which("nvidia-smi"):
        assert str(device) == "cuda:0"
    else:
        assert "cpu" in str(device).lower()


def test_torch_can_use_the_GPU():
    """Test that torch can use the GPU if it we have one."""

    assert torch.cuda.is_available() == bool(shutil.which("nvidia-smi"))


@pytest.fixture
def mock_train(monkeypatch: pytest.MonkeyPatch):
    mock_train_fn = Mock(spec=project.main.train)
    monkeypatch.setattr(project.main, project.main.train.__name__, mock_train_fn)
    return mock_train_fn


@pytest.fixture
def mock_evaluate_lightningmodule(monkeypatch: pytest.MonkeyPatch):
    mock_eval_lightningmodule = Mock(
        spec=project.main.evaluate_lightningmodule, return_value=("fake", 0.0, {})
    )
    monkeypatch.setattr(
        project.main, project.main.evaluate_lightningmodule.__name__, mock_eval_lightningmodule
    )
    return mock_eval_lightningmodule


@pytest.fixture
def mock_evaluate_jax_module(monkeypatch: pytest.MonkeyPatch):
    mock_eval_jax_module = Mock(
        spec=project.main.evaluate_jax_module, return_value=("fake", 0.0, {})
    )
    monkeypatch.setattr(
        project.main, project.main.evaluate_jax_module.__name__, mock_eval_jax_module
    )
    return mock_eval_jax_module


experiment_configs = [p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")]

experiment_commands_to_test = [
    "experiment=example trainer.fast_dev_run=True",
    "experiment=hf_example trainer.fast_dev_run=True",
    # "experiment=jax_example trainer.fast_dev_run=True",
    "experiment=jax_rl_example trainer.max_epochs=1",
    pytest.param(
        f"experiment=cluster_sweep_example "
        f"trainer/logger=[] "  # disable logging.
        f"trainer.fast_dev_run=True "  # make each job quicker to run
        f"hydra.sweeper.worker.max_trials=1 "  # limit the number of jobs that get launched.
        f"resources=gpu "
        f"cluster={'current' if SLURM_JOB_ID else 'mila'} ",
        marks=[
            pytest.mark.slow,
            pytest.mark.skipif(
                IN_GITHUB_CI,
                reason="Remote launcher tries to do a git push, doesn't work in github CI.",
            ),
            pytest.mark.xfail(
                raises=TypeError,
                reason="TODO: Getting a `TypeError: cannot pickle 'weakref.ReferenceType' object` error.",
                strict=False,
            ),
        ],
    ),
    pytest.param(
        "experiment=local_sweep_example "
        "trainer/logger=[] "  # disable logging.
        "trainer.fast_dev_run=True "  # make each job quicker to run
        "hydra.sweeper.worker.max_trials=2 ",  # Run a small number of trials.
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "experiment=profiling "
        "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
        "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
        "trainer.fast_dev_run=True ",  # make each job quicker to run
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "experiment=profiling "
        "algorithm=no_op "
        "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
        "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
        "trainer.fast_dev_run=True "  # make each job quicker to run
    ),
]


@pytest.mark.parametrize("experiment_config", experiment_configs)
def test_experiment_config_is_tested(experiment_config: str):
    select_experiment_command = f"experiment={experiment_config}"

    for test_command in experiment_commands_to_test:
        if isinstance(test_command, ParameterSet):
            assert len(test_command.values) == 1
            assert isinstance(test_command.values[0], str), test_command.values
            test_command = test_command.values[0]
        if select_experiment_command in test_command:
            return  # success.

    raise RuntimeError(f"{experiment_config=} is not tested by any of the test commands!")


@pytest.mark.parametrize(
    command_line_overrides.__name__,
    experiment_commands_to_test,
    indirect=True,
)
def test_can_load_experiment_configs(
    experiment_dictconfig: DictConfig,
    mock_train: Mock,
    mock_evaluate_lightningmodule: Mock,
    mock_evaluate_jax_module: Mock,
):
    # Mock out some part of the `main` function to not actually run anything.
    if experiment_dictconfig["hydra"]["mode"] == RunMode.MULTIRUN:
        # NOTE: Can't pass a dictconfig to `main` function when doing a multirun (seems to just do
        # a single run). If we try to call `main` without arguments and with the right arguments on\
        # the command-line, with the right functions mocked out, those might not get used at all
        # since `main` seems to create the launcher which pickles stuff and uses subprocesses.
        # Pretty gnarly stuff.
        pytest.skip(reason="Config is a multi-run config (e.g. a sweep). ")
    else:
        results = project.main.main(experiment_dictconfig)
        assert results is not None

    mock_train.assert_called_once()
    # One of them should have been called once.
    assert (mock_evaluate_lightningmodule.call_count == 1) ^ (
        mock_evaluate_jax_module.call_count == 1
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    command_line_overrides.__name__,
    experiment_commands_to_test,
    indirect=True,
)
def test_can_run_experiment(
    command_line_overrides: tuple[str, ...],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    # Mock out some part of the `main` function to not actually run anything.
    # Get a unique hash id:
    # todo: Set a unique name to avoid collisions between tests and reusing previous results.
    name = f"{request.function.__name__}_{uuid.uuid4().hex}"
    command_line_args = ["project/main.py"] + list(command_line_overrides) + [f"name={name}"]
    print(command_line_args)
    monkeypatch.setattr(sys, "argv", command_line_args)
    project.main.main()


@pytest.mark.parametrize(command_line_overrides.__name__, ["algorithm=example"], indirect=True)
def test_setting_just_algorithm_isnt_enough(experiment_dictconfig: DictConfig) -> None:
    """Test to check that the datamodule is required (even when just the example algorithm is set).

    TODO: We could probably move the `datamodule` config under `algorithm/datamodule`. Maybe that
    would be better?
    """
    with pytest.raises(
        omegaconf.errors.InterpolationResolutionError,
        match="Did you forget to set a value for the 'datamodule' config?",
    ):
        _ = resolve_dictconfig(experiment_dictconfig)


@pytest.mark.parametrize(
    command_line_overrides.__name__, ["algorithm=example datamodule=cifar10"], indirect=True
)
def test_example_experiment_defaults(experiment_config: Config) -> None:
    """Test to check that the datamodule is required (even when just an algorithm is set?!)."""

    assert experiment_config.algorithm["_target_"] == (
        ExampleAlgorithm.__module__ + "." + ExampleAlgorithm.__qualname__
    )
    assert (
        isinstance(experiment_config.datamodule, CIFAR10DataModule)
        or hydra_zen.get_target(experiment_config.datamodule) is CIFAR10DataModule
    )


@pytest.mark.skipif(
    IN_GITHUB_CI and sys.platform == "darwin",
    reason="TODO: Getting a 'MPS backend out of memory' error on the Github CI. ",
)
@pytest.mark.parametrize(
    command_line_overrides.__name__,
    [
        "algorithm=example datamodule=cifar10 seed=1 trainer/callbacks=none trainer.fast_dev_run=True"
    ],
    indirect=True,
)
def test_fast_dev_run(experiment_dictconfig: DictConfig):
    result = main(experiment_dictconfig)
    assert isinstance(result, dict)
    assert result["type"] == "objective"
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], float)


@pytest.mark.xfail(reason="TODO: cluster sweep example causes pydantic serialization error")
def test_run_auto_schema_via_cli_without_errors():
    """Checks that the command completes without errors."""
    # Run programmatically instead of with a subprocess so we can get nice coverage stats.
    # assuming we're at the project root directory.
    from hydra_auto_schema.__main__ import main as hydra_auto_schema_main

    hydra_auto_schema_main([str(CONFIG_DIR), "--stop-on-error", "-vv"])


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!

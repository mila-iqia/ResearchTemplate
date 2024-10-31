# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shutil
from unittest.mock import Mock

import hydra_zen
import omegaconf.errors
import pytest
import torch
from hydra.types import RunMode
from omegaconf import DictConfig

import project.main
from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.conftest import command_line_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID
from project.utils.hydra_utils import resolve_dictconfig

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
def mock_evaluate(monkeypatch: pytest.MonkeyPatch):
    mock_eval_fn = Mock(spec=project.main.evaluation, return_value=("fake", 0.0, {}))
    monkeypatch.setattr(project.main, project.main.evaluation.__name__, mock_eval_fn)
    return mock_eval_fn


experiment_configs = [p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")]


@pytest.mark.parametrize(
    command_line_overrides.__name__,
    [
        f"experiment={experiment}"
        if experiment != "cluster_sweep_example"
        else f"experiment={experiment} cluster=mila"
        if SLURM_JOB_ID is None
        else f"experiment={experiment} cluster=current"
        for experiment in list(experiment_configs)
    ],
    indirect=True,
    ids=[experiment for experiment in list(experiment_configs)],
)
def test_can_load_experiment_configs(
    experiment_dictconfig: DictConfig, mock_train: Mock, mock_evaluate: Mock
):
    # Mock out some part of the `main` function to not actually run anything.
    if experiment_dictconfig["hydra"]["mode"] == RunMode.MULTIRUN:
        pytest.skip(
            reason="Config is a multi-run config (e.g. a sweep). "
            "Not running with `main`, as it wouldn't make much sense."
        )
    results = project.main.main(experiment_dictconfig)
    assert results is not None
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()


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

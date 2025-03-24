# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import uuid
from logging import getLogger
from pathlib import Path
from unittest.mock import Mock

import omegaconf.errors
import pytest
import torch
from _pytest.mark.structures import ParameterSet
from hydra.types import RunMode
from omegaconf import DictConfig
from pytest_regressions.file_regression import FileRegressionFixture

import project.configs
import project.experiment
import project.main
from project.algorithms.no_op import NoOp
from project.conftest import setup_with_overrides, skip_on_macOS_in_CI
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_utils import resolve_dictconfig

logger = getLogger(__name__)

CONFIG_DIR = Path(project.configs.__file__).parent


experiment_configs = [p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")]
"""The list of all experiments configs in the `configs/experiment` directory.

This is used to check that all the experiment configs are covered by tests.
"""

experiment_commands_to_test: list[str | ParameterSet] = []
"""List of experiment commands to run for testing.

Consider adding a command that runs simple sanity check for your algorithm, something like one step
of training or something similar.
"""


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

    pytest.fail(
        f"Experiment config {experiment_config!r} is not covered by any of the tests!\n"
        f"Consider adding an example of an experiment command that uses this config to the "
        # This is a 'nameof' hack to get the name of the variable so we don't hard-code it.
        + ("`" + f"{experiment_commands_to_test=}".partition("=")[0] + "` list")
        + " list.\n"
        f"For example: 'experiment={experiment_config} trainer.max_epochs=1'."
    )


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
    mock_train_fn = Mock(spec=project.main.train, return_value=(None, None))
    monkeypatch.setattr(project.main, project.main.train.__name__, mock_train_fn)
    return mock_train_fn


@pytest.fixture
def mock_evaluate(monkeypatch: pytest.MonkeyPatch):
    mock_eval = Mock(spec=project.experiment.evaluate_lightning, return_value=("fake", 0.0, {}))
    monkeypatch.setattr(
        project.main,
        project.experiment.evaluate_lightning.__name__,
        mock_eval,
    )
    return mock_eval


@setup_with_overrides(experiment_commands_to_test)
def test_can_load_experiment_configs(
    dict_config: DictConfig,
    mock_train: Mock,
    mock_evaluate: Mock,
):
    # Mock out some part of the `main` function to not actually run anything.
    if dict_config["hydra"]["mode"] == RunMode.MULTIRUN:
        # NOTE: Can't pass a dictconfig to `main` function when doing a multirun (seems to just do
        # a single run). If we try to call `main` without arguments and with the right arguments on\
        # the command-line, with the right functions mocked out, those might not get used at all
        # since `main` seems to create the launcher which pickles stuff and uses subprocesses.
        # Pretty gnarly stuff.
        pytest.skip(reason="Config is a multi-run config (e.g. a sweep). ")
    else:
        results = project.main.main(dict_config)
        assert results is not None

    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()


@pytest.mark.slow
@setup_with_overrides(experiment_commands_to_test)
def test_can_run_experiment(
    command_line_overrides: tuple[str, ...],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Launches the sanity check experiments using the commands from the list above."""
    # Mock out some part of the `main` function to not actually run anything.
    # Get a unique hash id:
    # todo: Set a unique name to avoid collisions between tests and reusing previous results.
    name = f"{request.function.__name__}_{uuid.uuid4().hex}"
    command_line_args = ["project/main.py"] + list(command_line_overrides) + [f"name={name}"]
    logger.info(f"Launching sanity check experiment with command: {command_line_args}")
    monkeypatch.setattr(sys, "argv", command_line_args)
    project.main.main()


@skip_on_macOS_in_CI
@setup_with_overrides("algorithm=image_classifier")
def test_setting_just_algorithm_isnt_enough(dict_config: DictConfig) -> None:
    """Test to check that the datamodule is required (even when just the example algorithm is set).

    TODO: We could probably move the `datamodule` config under `algorithm/datamodule`. Maybe that
    would be better?
    """
    with pytest.raises(
        omegaconf.errors.InterpolationResolutionError,
        match="Did you forget to set a value for the 'datamodule' config?",
    ):
        _ = resolve_dictconfig(dict_config)


@pytest.mark.xfail(strict=False, reason="Regression files aren't necessarily present.")
def test_help_string(file_regression: FileRegressionFixture) -> None:
    help_string = subprocess.run(
        # Pass a seed so it isn't selected randomly, which would make the regression file change.
        shlex.split("python project/main.py seed=123 --help"),
        text=True,
        capture_output=True,
    ).stdout
    # Remove trailing whitespace so pre-commit doesn't change the regression file.
    # Also remove first or last empty lines (which would also be removed by pre-commit).
    help_string = "\n".join([line.rstrip() for line in help_string.splitlines()]).strip() + "\n"
    file_regression.check(help_string)


def test_run_auto_schema_via_cli_without_errors():
    """Checks that the command completes without errors."""
    # Run programmatically instead of with a subprocess so we can get nice coverage stats.
    # assuming we're at the project root directory.
    from hydra_auto_schema.__main__ import main as hydra_auto_schema_main

    hydra_auto_schema_main(
        [
            f"{REPO_ROOTDIR}",
            f"--configs_dir={CONFIG_DIR}",
            "--stop-on-error",
            "--regen-schemas",
            "-vv",
        ]
    )


@setup_with_overrides("algorithm=no_op trainer.max_epochs=1")
def test_setup_with_overrides_works(dict_config: omegaconf.DictConfig):
    """This test receives the `dict_config` loaded from Hydra with the given overrides."""
    assert dict_config["algorithm"]["_target_"] == NoOp.__module__ + "." + NoOp.__name__
    assert dict_config["trainer"]["max_epochs"] == 1


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!
# - Test offline mode for narval and such.

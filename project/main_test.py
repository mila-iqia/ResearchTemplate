# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import omegaconf.errors
import pytest
import torch
from hydra.types import RunMode
from omegaconf import DictConfig
from pytest_regressions.file_regression import FileRegressionFixture

import project.configs
import project.experiment
import project.main
from project.conftest import command_line_overrides, skip_on_macOS_in_CI
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.testutils import IN_GITHUB_CI

CONFIG_DIR = Path(project.configs.__file__).parent


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
    mock_eval = Mock(spec=project.experiment.evaluate, return_value=("fake", 0.0, {}))
    monkeypatch.setattr(
        project.main,
        project.experiment.evaluate.__name__,
        mock_eval,
    )
    return mock_eval


experiment_configs = [p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml")]
experiment_commands_to_test = [
    "experiment=example trainer.fast_dev_run=True",
    "experiment=text_classification_example trainer.fast_dev_run=True",
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
    (
        "experiment=profiling algorithm=no_op "
        "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
        "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
        "trainer.fast_dev_run=True "  # make each job quicker to run
    ),
    pytest.param(
        "experiment=llm_finetuning_example trainer.fast_dev_run=True trainer/logger=[]",
        marks=pytest.mark.skipif(
            SLURM_JOB_ID is None, reason="Can only be run on a slurm cluster."
        ),
    ),
]


@pytest.mark.parametrize(
    command_line_overrides.__name__,
    experiment_commands_to_test,
    indirect=True,
)
def test_can_load_experiment_configs(
    experiment_dictconfig: DictConfig,
    mock_train: Mock,
    mock_evaluate: Mock,
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
    mock_evaluate.assert_called_once()


@skip_on_macOS_in_CI
@pytest.mark.parametrize(
    command_line_overrides.__name__, ["algorithm=image_classifier"], indirect=True
)
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

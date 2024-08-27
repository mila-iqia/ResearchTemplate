# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shutil

import hydra_zen
import pytest
from omegaconf import DictConfig

from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.conftest import use_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule

from .main import main


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
    import torch

    assert torch.cuda.is_available() == bool(shutil.which("nvidia-smi"))


@pytest.mark.parametrize("overrides", [""], indirect=True)
def test_defaults(experiment_config: Config) -> None:
    """Test to check what the default values are when not specifying anything on the command-
    line."""
    assert experiment_config.algorithm["_target_"] == (
        ExampleAlgorithm.__module__ + "." + ExampleAlgorithm.__qualname__
    )
    assert (
        isinstance(experiment_config.datamodule, CIFAR10DataModule)
        or hydra_zen.get_target(experiment_config.datamodule) is CIFAR10DataModule
    )


@use_overrides(["seed=1 +trainer.fast_dev_run=True"])
def test_fast_dev_run(experiment_dictconfig: DictConfig):
    result = main(experiment_dictconfig)
    assert isinstance(result, dict)
    assert result["type"] == "objective"
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], float)


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!

# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import shutil

import hydra_zen

from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.conftest import use_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule


def test_jax_can_use_the_GPU():
    import jax.numpy

    device = jax.numpy.zeros(1).devices().pop()
    if shutil.which("nvidia-smi"):
        assert str(device) == "cuda:0"
    else:
        assert "cpu" in str(device).lower()


def test_torch_can_use_the_GPU():
    import torch

    assert torch.cuda.is_available() == bool(shutil.which("nvidia-smi"))


@use_overrides([""])
def test_defaults(experiment_config: Config) -> None:
    assert experiment_config.algorithm["_target_"] == (
        ExampleAlgorithm.__module__ + "." + ExampleAlgorithm.__name__
    )
    assert (
        isinstance(experiment_config.datamodule, CIFAR10DataModule)
        or hydra_zen.get_target(experiment_config.datamodule) is CIFAR10DataModule
    )


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!

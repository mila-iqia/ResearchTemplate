from pathlib import Path

import flax
import flax.linen
import pytest

from project.algorithms.jax_image_classifier import JaxImageClassifier
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.lightning_module_tests import LightningModuleTests


@pytest.fixture(autouse=True)
def prevent_jax_from_reserving_all_the_vram(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@run_for_all_configs_of_type("algorithm", JaxImageClassifier)
@run_for_all_configs_of_type("algorithm/network", flax.linen.Module)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
class TestJaxImageClassifier(LightningModuleTests[JaxImageClassifier]):
    """Tests for the Jax image classification algorithm.

    This simply reuses all the tests in the base test suite, specifying that the `datamodule`
    passed to the `JaxImageClassifier` should be for image classification and the `network` should be a
    `flax.linen.Module`.
    """


@pytest.mark.slow
def test_demo(tmp_path: Path):
    """Test the demo at the bottom of the module."""
    from .jax_image_classifier import demo

    demo(devices=1, overfit_batches=0.1, max_epochs=1, default_log_dir=tmp_path / "logs")

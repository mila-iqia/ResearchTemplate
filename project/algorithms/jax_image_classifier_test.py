from pathlib import Path
from typing import Any

import flax
import flax.linen
import pytest
from tensor_regression import TensorRegressionFixture

from project.algorithms.jax_image_classifier import JaxImageClassifier
from project.algorithms.testsuites.lightning_module_tests import GetStuffFromFirstTrainingStep
from project.conftest import fails_on_macOS_in_CI
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import IN_SELF_HOSTED_GITHUB_CI, run_for_all_configs_of_type

from .testsuites.lightning_module_tests import LightningModuleTests


@fails_on_macOS_in_CI
@run_for_all_configs_of_type("algorithm", JaxImageClassifier)
@run_for_all_configs_of_type("algorithm/network", flax.linen.Module)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
class TestJaxImageClassifier(LightningModuleTests[JaxImageClassifier]):
    """Tests for the Jax image classification algorithm.

    This simply reuses all the tests in the base test suite, specifying that the `datamodule`
    passed to the `JaxImageClassifier` should be for image classification and the `network` should be a
    `flax.linen.Module`.
    """

    @pytest.mark.xfail(
        IN_SELF_HOSTED_GITHUB_CI,
        reason="TODO: Test appears to be flaky only when run on the self-hosted runner?.",
    )
    def test_initialization_is_reproducible(
        self,
        training_step_content: tuple[
            JaxImageClassifier, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        return super().test_initialization_is_reproducible(
            training_step_content, tensor_regression, accelerator
        )


@pytest.mark.slow
def test_demo(tmp_path: Path):
    """Test the demo at the bottom of the module."""
    from .jax_image_classifier import demo

    demo(devices=1, overfit_batches=0.1, max_epochs=1, default_root_dir=tmp_path / "logs")

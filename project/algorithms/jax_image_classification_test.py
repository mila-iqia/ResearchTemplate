import flax
import flax.linen

from project.algorithms.jax_image_classifier import JaxImageClassifier
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.lightning_module_tests import LightningModuleTests


@run_for_all_configs_of_type("algorithm", JaxImageClassifier)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("network", flax.linen.Module)
class TestJaxImageClassifier(LightningModuleTests[JaxImageClassifier]):
    """Tests for the Jax image classification algorithm.

    This simply reuses all the tests in the base test suite, specifying that the `datamodule`
    passed to the ``JaxImageClassifier`` should be for image classification and the `network` should be a
    `flax.linen.Module`.
    """

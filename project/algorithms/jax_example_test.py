import flax
import flax.linen
import pytest

from project.algorithms.jax_example import JaxExample
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.algorithm_tests import LearningAlgorithmTests


@pytest.mark.timeout(10)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("network", flax.linen.Module)
class TestJaxExample(LearningAlgorithmTests[JaxExample]): ...

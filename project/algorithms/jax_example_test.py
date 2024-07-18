import flax
import flax.linen

from project.algorithms.jax_example import JaxExample
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.algorithm_tests import LearningAlgorithmTests


@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("network", flax.linen.Module)
class TestJaxExample(LearningAlgorithmTests[JaxExample]): ...

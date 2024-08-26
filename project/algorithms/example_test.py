"""Example showing how the test suite can be used to add tests for a new algorithm."""

import torch.nn

from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .example import ExampleAlgorithm


@run_for_all_configs_of_type("algorithm", ExampleAlgorithm)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("network", torch.nn.Module)
class TestExampleAlgo(LearningAlgorithmTests[ExampleAlgorithm]):
    """Tests for the `ExampleAlgorithm`.

    This runs all the tests included in the base class, with the given parametrizations:
    - `algorithm_config` will take the value


    See [LearningAlgorithmTests][project.algorithms.testsuites.algorithm_tests.LearningAlgorithmTests]
    for more information on the built-in tests.
    """

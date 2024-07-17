import torch

from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .example import ExampleAlgorithm


@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("network", torch.nn.Module)
class TestExampleAlgo(LearningAlgorithmTests[ExampleAlgorithm]):
    """Tests for the `ExampleAlgorithm`.

    See `LearningAlgorithmTests` for more information on the built-in tests.
    """

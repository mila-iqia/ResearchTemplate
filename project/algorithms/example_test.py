"""Example showing how the test suite can be used to add tests for a new algorithm."""

import pytest
import torch
from transformers import PreTrainedModel

from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.hydra_config_utils import get_all_configs_in_group_of_type
from project.utils.testutils import run_for_all_configs_of_type

from .example import ExampleAlgorithm


@run_for_all_configs_of_type("algorithm", ExampleAlgorithm)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@pytest.mark.parametrize(
    "overrides",
    [
        f"algorithm/network={config}"
        for config in get_all_configs_in_group_of_type(
            config_group="algorithm/network",
            config_target_type=torch.nn.Module,
            include_subclasses=True,
            excluding=PreTrainedModel,
        )
    ],
    indirect=True,
)
class TestExampleAlgo(LearningAlgorithmTests[ExampleAlgorithm]):
    """Tests for the `ExampleAlgorithm`.

    This runs all the tests included in the base class, with the given parametrizations:

    - `algorithm_config` will take the value `"example"`
        - This is because there is an `example.yaml` config file whose `_target_` is the ``ExampleAlgorithm``.
    - `datamodule_config` will take these values: `['cifar10', 'fashion_mnist', 'imagenet', 'imagenet32', 'inaturalist', 'mnist']`
        - These are all the configs whose target is an `ImageClassificationDataModule`.
    - Similarly, `network_config` will be parametrized by the names of all configs which produce an nn.Module.

    Take a look at the [LearningAlgorithmTests class][project.algorithms.testsuites.algorithm_tests.LearningAlgorithmTests]
    if you want to see the actual test code.
    """

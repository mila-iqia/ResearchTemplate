"""Example showing how the test suite can be used to add tests for a new algorithm."""

import pytest
import torch
from transformers import PreTrainedModel

from project.algorithms.testsuites.lightning_module_tests import LightningModuleTests
from project.configs import Config
from project.conftest import command_line_overrides, skip_on_macOS_in_CI
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.testutils import run_for_all_configs_of_type

from .image_classifier import ImageClassifier


@pytest.mark.parametrize(
    command_line_overrides.__name__,
    ["algorithm=image_classifier datamodule=cifar10"],
    indirect=True,
)
def test_example_experiment_defaults(experiment_config: Config) -> None:
    """Test to check that the datamodule is required (even when just an algorithm is set?!)."""

    assert experiment_config.algorithm["_target_"] == (
        ImageClassifier.__module__ + "." + ImageClassifier.__qualname__
    )

    assert isinstance(experiment_config.datamodule, CIFAR10DataModule)


@skip_on_macOS_in_CI
@run_for_all_configs_of_type("algorithm", ImageClassifier)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("algorithm/network", torch.nn.Module, excluding=PreTrainedModel)
class TestImageClassifier(LightningModuleTests[ImageClassifier]):
    """Tests for the `ImageClassifier`.

    This runs all the tests included in the base class, with the given parametrizations:

    - `algorithm_config` will take the value `"image_classifier"`
        - This is because there is an `image_classifier.yaml` config file in project/configs/algorithms
          whose `_target_` is the `ImageClassifier`.
    - `datamodule_config` will take these values: `['cifar10', 'fashion_mnist', 'imagenet', 'inaturalist', 'mnist']`
        - These are all the configs whose target is an `ImageClassificationDataModule`.
    - Similarly, `network_config` will be parametrized by the names of all configs which produce an nn.Module,
      except those that would create a `PreTrainedModel` from HuggingFace.
        - This is currently the easiest way for us to say "any network for image classification.

    Take a look at the `LightningModuleTests` class if you want to see the actual test code.
    """

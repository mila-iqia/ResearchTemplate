"""Example showing how the test suite can be used to add tests for a new algorithm."""

import pytest
import torch

from project.algorithms.testsuites.lightning_module_tests import LightningModuleTests
from project.configs import Config
from project.conftest import setup_with_overrides, skip_on_macOS_in_CI
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.main_test import experiment_commands_to_test
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.testutils import IN_GITHUB_CI, run_for_all_configs_of_type

from .image_classifier import ImageClassifier

experiment_commands_to_test.extend(
    [
        "experiment=example trainer.fast_dev_run=True",
        pytest.param(
            f"experiment=cluster_sweep_example "
            f"trainer/logger=[] "  # disable logging.
            f"trainer.fast_dev_run=True "  # make each job quicker to run
            f"hydra.sweeper.worker.max_trials=1 "  # limit the number of jobs that get launched.
            f"resources=gpu "
            f"cluster={'current' if SLURM_JOB_ID else 'mila'} ",
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    IN_GITHUB_CI,
                    reason="Remote launcher tries to do a git push, doesn't work in github CI.",
                ),
            ],
        ),
        pytest.param(
            "experiment=local_sweep_example "
            "trainer/logger=[] "  # disable logging.
            "trainer.fast_dev_run=True "  # make each job quicker to run
            "hydra.sweeper.worker.max_trials=2 ",  # Run a small number of trials.
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "experiment=profiling "
            "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
            "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
            "trainer.fast_dev_run=True ",  # make each job quicker to run
            marks=pytest.mark.slow,
        ),
        (
            "experiment=profiling algorithm=no_op "
            "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
            "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
            "trainer.fast_dev_run=True "  # make each job quicker to run
        ),
    ]
)


@setup_with_overrides("algorithm=image_classifier datamodule=cifar10")
def test_example_experiment_defaults(config: Config) -> None:
    """Test to check that the datamodule is required (even when just an algorithm is set?!)."""

    assert config.algorithm["_target_"] == (
        ImageClassifier.__module__ + "." + ImageClassifier.__qualname__
    )

    assert isinstance(config.datamodule, CIFAR10DataModule)


# When the `transformers` library is installed, for example when NLP-related examples are included,
# then we don't want this "run for all subclasses of nn.Module" to match these NLP models.
try:
    from transformers import PreTrainedModel

    excluding = PreTrainedModel
except ImportError:
    excluding = ()


@skip_on_macOS_in_CI
@run_for_all_configs_of_type("algorithm", ImageClassifier)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("algorithm/network", torch.nn.Module, excluding=excluding)
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

"""Unit tests for the llm finetuning example."""

import copy
import operator

import jax
import lightning
import pytest
import torch
from tensor_regression import TensorRegressionFixture
from tensor_regression.stats import get_simple_attributes
from torch.utils.data import DataLoader

from project.algorithms.hf_llm_finetuning_example import DatasetConfig
from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.configs.config import Config
from project.utils.testutils import run_for_all_configs_of_type
from project.utils.typing_utils import PyTree
from project.utils.typing_utils.protocols import DataModule

from .llm_finetuning_example import LLMFinetuningExample, get_hash_of


def test_get_hash_of():
    d1 = DatasetConfig(dataset_path="wikitext", dataset_name="wikitext-2-v1")
    d2 = DatasetConfig(dataset_path="wikitext", dataset_name="wikitext-103-v1")
    assert get_hash_of(d1) == get_hash_of(d1)
    assert get_hash_of(d1) == get_hash_of(copy.deepcopy(d1))
    assert get_hash_of(d1) != get_hash_of(d2)


@get_simple_attributes.register(tuple)
def _get_tuple_attributes(value: tuple, precision: int | None):
    # This is called to get some simple stats to store in regression files during tests, in
    # particular for tuples (since there isn't already a handler for it in the tensor_regression
    # package.)
    # Note: This information about this output is not very descriptive.
    # not this is called only for the `out.past_key_values` entry in the `CausalLMOutputWithPast`
    # that is returned from the forward pass output.
    num_items_to_include = 5  # only show the stats of some of the items.
    return {
        "length": len(value),
        **{
            f"{i}": get_simple_attributes(item, precision=precision)
            for i, item in enumerate(value[:num_items_to_include])
        },
    }


@run_for_all_configs_of_type("algorithm", LLMFinetuningExample)
class TestLLMFinetuningExample(LearningAlgorithmTests[LLMFinetuningExample]):
    # TODO: annoying that we need to redefine this here and change its scope, since we get the
    # dataloader from the module and not from a datamodule (since this example does not use a datamodule).
    @pytest.fixture(scope="function")
    def train_dataloader(
        self, algorithm: LLMFinetuningExample, request: pytest.FixtureRequest
    ) -> DataLoader:
        algorithm.prepare_data()
        algorithm.setup("fit")

        train_dataloader = algorithm.train_dataloader()
        assert isinstance(train_dataloader, DataLoader)
        return train_dataloader

    @pytest.fixture(scope="function")
    def training_batch(
        self, train_dataloader: DataLoader, device: torch.device
    ) -> dict[str, torch.Tensor]:
        # Get a batch of data from the dataloader.

        # The batch of data will always be the same because the dataloaders are passed a Generator
        # object in their constructor.
        assert isinstance(train_dataloader, DataLoader)
        dataloader_iterator = iter(train_dataloader)

        with torch.random.fork_rng(list(range(torch.cuda.device_count()))):
            # TODO: This ugliness is because torchvision transforms use the global pytorch RNG!
            torch.random.manual_seed(42)
            batch = next(dataloader_iterator)

        return jax.tree.map(operator.methodcaller("to", device=device), batch)

    @pytest.fixture(scope="function")
    def forward_pass_input(self, training_batch: PyTree[torch.Tensor], device: torch.device):
        """Extracts the model input from a batch of data coming from the dataloader.

        Overwrite this if your batches are not tuples of tensors (i.e. if your algorithm isn't a
        simple supervised learning algorithm like the example).
        """
        assert isinstance(training_batch, dict)
        return training_batch

    # Checking all the weights against the 900mb reference .npz file is a bit slow.
    @pytest.mark.slow
    def test_initialization_is_reproducible(
        self,
        experiment_config: Config,
        datamodule: DataModule,
        seed: int,
        tensor_regression: TensorRegressionFixture,
        trainer: lightning.Trainer,
    ):
        super().test_initialization_is_reproducible(
            experiment_config=experiment_config,
            datamodule=datamodule,
            seed=seed,
            tensor_regression=tensor_regression,
            trainer=trainer,
        )

"""Unit tests for the llm finetuning example."""

import operator

import jax
import pytest
import torch
from torch.utils.data import DataLoader

from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.utils.testutils import run_for_all_configs_of_type
from project.utils.typing_utils import PyTree

from .llm_finetuning_example import LLMFinetuningExample


def test_get_hash_of(): ...


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

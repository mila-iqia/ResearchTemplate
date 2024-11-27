"""Unit tests for the llm finetuning example."""

import copy
import operator
from typing import Any

import jax
import lightning
import pytest
import torch
from tensor_regression import TensorRegressionFixture
from torch.utils.data import DataLoader

from project.algorithms.llm_finetuning import (
    DatasetConfig,
    LLMFinetuningExample,
    TokenizerConfig,
    get_hash_of,
)
from project.algorithms.testsuites.lightning_module_tests import LightningModuleTests
from project.configs.config import Config
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.testutils import run_for_all_configs_of_type, total_vram_gb
from project.utils.typing_utils import PyTree


@pytest.mark.parametrize(
    ("c1", "c2"),
    [
        (
            DatasetConfig(dataset_path="wikitext", dataset_name="wikitext-2-v1"),
            DatasetConfig(dataset_path="wikitext", dataset_name="wikitext-103-v1"),
        ),
        (
            TokenizerConfig(pretrained_model_name_or_path="gpt2"),
            TokenizerConfig(pretrained_model_name_or_path="bert-base-uncased"),
        ),
    ],
)
def test_get_hash_of(c1, c2):
    assert get_hash_of(c1) == get_hash_of(c1)
    assert get_hash_of(c2) == get_hash_of(c2)
    assert get_hash_of(c1) != get_hash_of(c2)
    assert get_hash_of(c1) == get_hash_of(copy.deepcopy(c1))
    assert get_hash_of(c2) == get_hash_of(copy.deepcopy(c2))


@pytest.mark.skipif(total_vram_gb() < 16, reason="Not enough VRAM to run this test.")
@run_for_all_configs_of_type("algorithm", LLMFinetuningExample)
class TestLLMFinetuningExample(LightningModuleTests[LLMFinetuningExample]):
    @pytest.fixture(scope="function")
    def train_dataloader(
        self,
        algorithm: LLMFinetuningExample,
        request: pytest.FixtureRequest,
        trainer: lightning.Trainer,
    ) -> DataLoader:
        """Fixture that creates and returns the training dataloader.

        NOTE: Here we're purpusefully redefining the `project.conftest.train_dataloader` fixture
        because it assumes that the algorithm uses a datamodule.
        Here we change the fixture scope.
        """
        # a bit hacky: Set the trainer on the lightningmodule.
        algorithm._trainer = trainer
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
            # torch.random.manual_seed(42)
            lightning.seed_everything(42, workers=True)
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

    def test_training_batch_doesnt_change(
        self, training_batch: dict, tensor_regression: TensorRegressionFixture
    ):
        tensor_regression.check(training_batch, include_gpu_name_in_stats=False)

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
    )
    @pytest.mark.slow  # Checking against the 900mb reference .npz file is a bit slow.
    def test_initialization_is_reproducible(
        self,
        experiment_config: Config,
        datamodule: lightning.LightningDataModule,
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

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
    )
    def test_forward_pass_is_reproducible(
        self,
        forward_pass_input: Any,
        algorithm: LLMFinetuningExample,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        return super().test_forward_pass_is_reproducible(
            forward_pass_input=forward_pass_input,
            algorithm=algorithm,
            seed=seed,
            tensor_regression=tensor_regression,
        )

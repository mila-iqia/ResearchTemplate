"""Unit tests for the llm finetuning example."""

import copy
from typing import Any

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
from project.algorithms.testsuites.lightning_module_tests import (
    GetStuffFromFirstTrainingStep,
    LightningModuleTests,
)
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.testutils import run_for_all_configs_of_type, total_vram_gb


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
    @pytest.fixture(scope="class")
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
        with torch.random.fork_rng(list(range(torch.cuda.device_count()))):
            # TODO: This is necessary because torchvision transforms use the global pytorch RNG!
            lightning.seed_everything(42, workers=True)

            algorithm.prepare_data()
            algorithm.setup("fit")

        train_dataloader = algorithm.train_dataloader()
        assert isinstance(train_dataloader, DataLoader)
        return train_dataloader

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
    )
    @pytest.mark.slow  # Checking against the 900mb reference .npz file is a bit slow.
    def test_initialization_is_reproducible(
        self,
        training_step_content: tuple[
            LLMFinetuningExample, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        super().test_initialization_is_reproducible(
            training_step_content=training_step_content,
            tensor_regression=tensor_regression,
            accelerator=accelerator,
        )

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
    )
    def test_forward_pass_is_reproducible(
        self,
        training_step_content: tuple[
            LLMFinetuningExample, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
    ):
        return super().test_forward_pass_is_reproducible(
            training_step_content=training_step_content, tensor_regression=tensor_regression
        )

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
    )
    def test_backward_pass_is_reproducible(
        self,
        training_step_content: tuple[
            LLMFinetuningExample, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        return super().test_backward_pass_is_reproducible(
            training_step_content=training_step_content,
            tensor_regression=tensor_regression,
            accelerator=accelerator,
        )

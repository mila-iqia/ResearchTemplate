"""Unit tests for the llm finetuning example."""

import copy

import pytest

from project.algorithms.llm_finetuning import (
    DatasetConfig,
    LLMFinetuningExample,
    TokenizerConfig,
    get_hash_of,
)
from project.algorithms.testsuites.lightning_module_tests import (
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


@pytest.mark.xfail(
    SLURM_JOB_ID is not None, reason="TODO: Seems to be failing when run on a SLURM cluster."
)
@pytest.mark.slow  # Checking against the 900mb reference .npz file is a bit slow.
@pytest.mark.skipif(total_vram_gb() < 16, reason="Not enough VRAM to run this test.")
@run_for_all_configs_of_type("algorithm", LLMFinetuningExample)
class TestLLMFinetuningExample(LightningModuleTests[LLMFinetuningExample]):
    """Tests for the LLM fine-tuning example."""

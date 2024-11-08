import dataclasses
import itertools
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import datasets
import hydra_zen
import torch
from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict
from hydra_zen.typing import Builds
from lightning import LightningModule
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase

from project.utils.env_vars import SCRATCH, SLURM_TMPDIR
from project.utils.utils import default_device

logger = getLogger(__name__)


def num_cpus_per_task() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


@hydra_zen.hydrated_dataclass(
    target=AutoModelForCausalLM.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class NetworkConfig(Builds[type[AutoModelForCausalLM]]):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    cache_dir: Path | None = None
    force_download: bool | None = None
    local_files_only: bool | None = None
    proxies: dict[str, str] | None = None
    revision: str | None = None
    subfolder: str | None = None
    use_auth_token: bool | None = None
    token: str | bool | None = None


@hydra_zen.hydrated_dataclass(
    target=AutoTokenizer.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TokenizerConfig:
    pretrained_model_name_or_path: str
    cache_dir: Path | None = None  # use standard cache by default.
    force_download: bool = False
    local_files_only: bool = False
    token: str | bool | None = None
    revision: str = "main"
    use_fast: bool = True
    config: PretrainedConfig | None = None
    proxies: dict[str, str] = dataclasses.field(default_factory=dict)
    subfolder: str = ""
    tokenizer_type: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True, unsafe_hash=True)
class DatasetConfig:
    """Configuration options related to the choice of dataset."""

    dataset_path: str
    """Name of the dataset "family"?

    For example, to load "wikitext/wikitext-103-v1", this would be "wikitext".
    """

    dataset_name: str | None = None
    """Name of the specific dataset?

    For example, to load "wikitext/wikitext-103-v1", this would be "wikitext-103-v1".
    """

    per_device_eval_batch_size: int = 4
    per_device_train_batch_size: int = 2

    block_size: int = 1024

    preprocessing_num_workers: int = num_cpus_per_task()

    validation_split_percentage: int = 10
    """Fraction of the train dataset to use for validation if there isn't already a validation
    split."""

    overwrite_cache: bool = False


def load_raw_datasets(config: DatasetConfig):
    raw_datasets = datasets.load_dataset(config.dataset_path, config.dataset_name)
    assert isinstance(raw_datasets, DatasetDict)
    if "validation" not in raw_datasets.keys() and config.validation_split_percentage > 0:
        raw_datasets["validation"] = datasets.load_dataset(
            config.dataset_path,
            config.dataset_name,
            split=f"train[:{config.validation_split_percentage}%]",
        )
        raw_datasets["train"] = datasets.load_dataset(
            config.dataset_path,
            config.dataset_name,
            split=f"train[{config.validation_split_percentage}%:]",
        )
    return raw_datasets


def load_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizerBase:
    return hydra_zen.instantiate(config)


def tokenize_datasets(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DatasetConfig,
) -> DatasetDict:
    return raw_datasets.map(
        lambda b: tokenizer(b["text"]),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=not config.overwrite_cache,
        desc="Running tokenizer on dataset",
    )


def group_text_into_blocks(
    tokenized_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DatasetConfig,
) -> DatasetDict:
    block_size = config.block_size
    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
        block_size = tokenizer.model_max_length

    return tokenized_datasets.map(
        group_texts,
        fn_kwargs={"block_size": block_size},
        batched=True,
        load_from_cache_file=True,
        num_proc=config.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples: dict, block_size: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


class LLMFinetuningExample(LightningModule):
    """Example of a lightning module used to train a huggingface model."""

    def __init__(
        self,
        network_config: NetworkConfig,
        tokenizer_config: TokenizerConfig,
        dataset_config: DatasetConfig,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        init_seed: int = 42,
    ):
        super().__init__()
        self.network_config = network_config
        self.tokenizer_config = tokenizer_config
        self.dataset_config = dataset_config
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.init_seed = init_seed

        with default_device(), torch.random.fork_rng():
            # deterministic weight initialization
            # Initializes the weights on the GPU if we have one, so we don't request lots of RAM
            # just to load up the model weights and then not use it.
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)

        # Small fix for the `device` property in LightningModule, which is CPU by default.
        self._device = next((p.device for p in self.parameters()), torch.device("cpu"))

        self.save_hyperparameters(ignore=["network", "datamodule"])
        self.prepare_data_per_node = True  # let Pytorch-Lightning know about this.

        self.unique_name = f"{hash(self.dataset_config)}_{hash(self.tokenizer_config)}"
        logger.info(f"Unique name for our dataset / tokenizer configs: {self.unique_name}")
        # TODO: Currently, we preprocess the dataset and save it in $SLURM_TMPDIR.
        # If we restart after preemption, we have re-tokenize the dataset, which sucks.
        # Instead, we could prepare the dataset once and save it to $SCRATCH.
        # Then, we could copy it to $SLURM_TMPDIR and then load from
        # $SLURM_TMPDIR in `setup`.
        self.processed_dataset_dir = (SLURM_TMPDIR or Path.cwd()) / "data"

        # TODO: Should we base ourselves on the HF environment variables instead?
        assert SCRATCH is not None, "TODO: figure out where to put this otherwise."
        self.scratch_prepared_dataset_dir = (
            SCRATCH / "data" / "prepared_dataset" / self.unique_name
        )
        self.scratch_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        assert SLURM_TMPDIR is not None, "TODO: figure out where to put this otherwise."
        self.tmpdir_prepared_dataset_dir = (
            SLURM_TMPDIR / "data" / "prepared_dataset" / self.unique_name
        )
        self.tmpdir_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        # Load the tokenizer.
        _ = load_tokenizer(self.tokenizer_config)

        # If we've already prepared the dataset on this node, we can just load it.
        # TODO: we might need to use a custom file name or pass some args like block size so we can
        # 'invalidate the cache' when we change the block size or the tokenizer, etc?

        try:
            load_from_disk(self.processed_dataset_dir)
            return
        except FileNotFoundError:
            pass
        # Otherwise do the tokenization and grouping and save to a file so we can just load it in
        # `setup`.
        raw_datasets = load_raw_datasets(self.dataset_config)
        tokenized_datasets = tokenize_datasets(raw_datasets, self.tokenizer, self.dataset_config)
        lm_datasets = group_text_into_blocks(
            tokenized_datasets, self.tokenizer, self.dataset_config
        )

        lm_datasets.save_to_disk()
        logger.info(f"Saved processed dataset to {self.processed_dataset_dir}")

    def setup(self, stage: str):
        # Load the tokenizer (again).
        # This is done here again because in distributed training jobs, `prepare_data` is only
        # called in the first task on each node, while `setup` is called in every task.
        self.tokenizer = load_tokenizer(self.tokenizer_config)
        logger.info(f"Loading processed dataset from {self.processed_dataset_dir}")
        lm_datasets = load_from_disk(self.processed_dataset_dir)
        assert isinstance(lm_datasets, DatasetDict)
        self.train_dataset = lm_datasets["train"]
        self.valid_dataset = lm_datasets["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.per_device_train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=default_data_collator,
            batch_size=self.per_device_eval_batch_size,
        )

    def forward(self, **inputs: torch.Tensor) -> BaseModelOutput:
        return self.network(**inputs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs: CausalLMOutput | SequenceClassifierOutput = self(**batch)
        loss = outputs.loss
        assert isinstance(loss, torch.Tensor), loss
        # todo: log more stuff!
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        outputs: CausalLMOutput | SequenceClassifierOutput = self(**batch)
        loss = outputs.loss
        assert isinstance(loss, torch.Tensor)
        # todo: log the output of the metric.
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if isinstance(outputs, SequenceClassifierOutput):
            metric_value = self.metric.compute(
                predictions=outputs.logits, references=batch["labels"]
            )
            assert False, metric_value
            self.log(
                f"train/{self.hf_metric_name}",
                metric_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.network
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd_param in n for nd_param in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd_param in n for nd_param in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

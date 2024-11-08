import functools
import itertools
import os
from logging import getLogger
from pathlib import Path

import datasets
import torch
from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict
from lightning import LightningModule
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase

from project.utils.env_vars import SLURM_TMPDIR

logger = getLogger(__name__)


def num_cpus_per_task() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


class LLMFinetuningExample(LightningModule):
    """Example of a lightning module used to train a huggingface model."""

    def __init__(
        self,
        network: PreTrainedModel | functools.partial[PreTrainedModel],
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str,
        dataset_name: str | None = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        per_device_eval_batch_size: int = 4,
        per_device_train_batch_size: int = 2,
        block_size: int = 1024,
        preprocessing_num_workers: int = num_cpus_per_task(),
        init_seed: int = 42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.init_seed = init_seed
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.tokenizer = tokenizer
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.per_device_train_batch_size = per_device_train_batch_size
        self.preprocessing_num_workers = preprocessing_num_workers

        self.processed_dataset_dir = (SLURM_TMPDIR or Path.cwd()) / "data"

        with torch.random.fork_rng():
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            if isinstance(network, functools.partial):
                network = network()
            self.network = network

        # Small fix for the `device` property in LightningModule, which is CPU by default.
        self._device = next((p.device for p in self.parameters()), torch.device("cpu"))

        if block_size > self.tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
            )
            block_size = tokenizer.model_max_length
        self.block_size = block_size
        self.save_hyperparameters(ignore=["network", "datamodule"])

    def _load_datasets(self):
        raw_datasets = datasets.load_dataset(self.dataset_path, self.dataset_name)
        assert isinstance(raw_datasets, DatasetDict)
        if "validation" not in raw_datasets.keys():
            validation_split_percentage = 10
            raw_datasets["validation"] = datasets.load_dataset(
                self.dataset_path,
                self.dataset_name,
                split=f"train[:{validation_split_percentage}%]",
            )
            raw_datasets["train"] = datasets.load_dataset(
                self.dataset_path,
                self.dataset_name,
                split=f"train[{validation_split_percentage}%:]",
            )
        return raw_datasets

    def _tokenize_datasets(self, raw_datasets: DatasetDict) -> DatasetDict:
        return raw_datasets.map(
            lambda b: self.tokenizer(b["text"]),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    def _group_texts(self, tokenized_datasets: DatasetDict) -> DatasetDict:
        return tokenized_datasets.map(
            group_texts,
            fn_kwargs={"block_size": self.block_size},
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {self.block_size}",
        )

    def prepare_data(self):
        # If we've already prepared the dataset on this node, we can just load it.
        # TODO: we might need to use a custom file name or pass some args like block size so we can
        # 'invalidate the cache' when we change the block size or the tokenizer, etc?
        try:
            load_from_disk(self.processed_dataset_dir, keep_in_memory=True)
            return
        except FileNotFoundError:
            pass
        # Otherwise do the tokenization and grouping and save to a file so we can just load it in
        # `setup`.
        raw_datasets = self._load_datasets()
        tokenized_datasets = self._tokenize_datasets(raw_datasets)
        lm_datasets = self._group_texts(tokenized_datasets)
        lm_datasets.save_to_disk(self.processed_dataset_dir)
        logger.info(f"Saved processed dataset to {self.processed_dataset_dir}")

    def setup(self, stage: str):
        logger.info(f"Loading processed dataset from {self.processed_dataset_dir}")
        lm_datasets = load_from_disk(self.processed_dataset_dir)
        assert isinstance(lm_datasets, DatasetDict)
        # raw_datasets = self._load_datasets()
        # tokenized_datasets = self._tokenize_datasets(raw_datasets)
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

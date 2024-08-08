from __future__ import annotations

import os
import shutil

from datasets import DatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)

SCRATCH = os.environ.get("SCRATCH")
SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR")


class GLUEDataModule(LightningDataModule):  ## to be homogenized with the base text class
    """
    Lightning data module for GLUE datasets.
    based off of https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/text-transformers.html
    """

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        self.model_name_or_path = model_name_or_path
        self.dataset_path = os.path.join(SCRATCH, f"{self.task_name}_dataset")
        self.tmp_path = os.path.join(SLURM_TMPDIR, f"{self.task_name}_tmp")
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True, cache_dir=SCRATCH
        )

    def prepare_data(self):
        # Make sure to use $SCRATCH instead of $HOME for the huggingface cache directory.
        if not os.path.exists(self.dataset_path):
            dataset = load_dataset("glue", self.task_name, cache_dir=SCRATCH)
            # Tokenize and save to $SCRATCH
            tokenized_dataset = dataset.map(self.convert_to_features, batched=True)
            tokenized_dataset.save_to_disk(self.dataset_path)
        else:
            print(f"Tokenized dataset already exists at {self.dataset_path}")

    def setup(self, stage: str):
        if not os.path.exists(self.tmp_path):
            # Copy to $SLURM_TMPDIR before being loaded
            shutil.copytree(self.dataset_path, self.tmp_path)

        self.dataset = DatasetDict.load_from_disk(self.tmp_path)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

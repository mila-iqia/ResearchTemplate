"""Example algorithm that can train a huggingface model.

Also check out this link for more detailed example script:

https://github.com/lebrice/mila-docs/blob/llm_training/docs/examples/distributed/LLM_training/main.py
"""

from __future__ import annotations

import shutil
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from project.utils.env_vars import REPO_ROOTDIR, SCRATCH, SLURM_TMPDIR

# BUG: Investigating a slowdown that is happening on SLURM clusters with Lightning and this datamodule:
# https://github.com/Lightning-AI/pytorch-lightning/issues/10389#issuecomment-2310630247
# torch.set_num_threads(1)

logger = getLogger(__name__)

SupportedTask = Literal["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli", "ax"]


task_field_map = {
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

num_labels = {
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


class TextClassificationDataModule(LightningDataModule):
    """Lightning data module for HF text classification datasets.

    This is based on this tutorial:
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/text-transformers.html
    """

    def __init__(
        self,
        hf_dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        task_name: str,
        text_fields: list[str] | None = None,
        num_classes: int | None = None,
        data_dir: str | Path = SCRATCH or REPO_ROOTDIR / "data",
        loader_columns: list = [
            "datasets_idx",
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
            "labels",
        ],
        seed: int = 42,
        shuffle: bool = True,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        # use for debugging; NOT STABLE, may cause memory allocation issues
        dataset_fraction: float | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.loader_columns = loader_columns
        self.seed = seed
        self.shuffle = shuffle
        self.hf_dataset_path = hf_dataset_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_fraction = dataset_fraction
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_dir: Path = Path(data_dir)
        self.processed_dataset_path = (
            self.data_dir / f"{self.hf_dataset_path}_{self.task_name}_dataset"
        )

        if text_fields is None:
            text_fields = task_field_map.get(task_name)
        self.text_fields = text_fields or ["text"]

        if num_classes is None:
            num_classes = num_labels.get(task_name)
        self.num_classes = num_classes

        if SLURM_TMPDIR:
            self.working_path = SLURM_TMPDIR / self.processed_dataset_path.name
        else:
            self.working_path = self.processed_dataset_path

        ## todo: verify authentication method setup. Is trust_remote_code the right play here?
        _rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.train_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())
        self.val_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())
        self.test_dl_rng_seed = int(torch.randint(0, int(1e6), (1,), generator=_rng).item())

    def prepare_data(self):
        # Make sure to use $SCRATCH instead of $HOME for the huggingface cache directory"
        logger.debug("Loading dataset...")
        dataset = load_dataset(
            self.hf_dataset_path,
            self.task_name,
            cache_dir=str(self.data_dir / ".cache/huggingface/datasets"),
            save_infos=True,
        )
        # Tokenize and save to $SCRATCH
        tokenized_dataset = dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=(["label"] if "label" in dataset.column_names else []),
            load_from_cache_file=True,
        )
        logger.debug(f"Saving (overwriting) tokenized dataset at {self.processed_dataset_path}")
        tokenized_dataset.save_to_disk(str(self.processed_dataset_path))

        # Copy dataset to the (faster) temporary path if not already present
        if self.working_path != self.processed_dataset_path and not self.working_path.exists():
            logger.debug(f"Copying dataset from {self.working_path} to {self.working_path}")
            shutil.copytree(self.processed_dataset_path, self.working_path, dirs_exist_ok=False)

        logger.info(f"Done preparing the dataset at {self.processed_dataset_path}")

    def setup(self, stage: str, seed: int = 42):
        self.dataset = DatasetDict.load_from_disk(self.working_path)
        logger.info(f"Loaded dataset from {self.working_path}")

        if self.dataset_fraction is not None:
            logger.info(f"Reducing dataset to {self.dataset_fraction * 100}% of original size")
            self.dataset = self._apply_dataset_fraction(self.dataset, self.dataset_fraction, seed)

        for split in self.dataset.keys():
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            logger.info(f"Setting format for {split} split")
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        logger.info(f"Setup complete for {self.task_name}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            generator=torch.Generator().manual_seed(self.train_dl_rng_seed),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.val_dl_rng_seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    generator=torch.Generator().manual_seed(self.val_dl_rng_seed + i),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
                for i, x in enumerate(self.eval_splits)
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["test"],
                batch_size=self.eval_batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.test_dl_rng_seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    generator=torch.Generator().manual_seed(self.test_dl_rng_seed + i),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
                for i, x in enumerate(self.eval_splits)
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
        if "label" in example_batch and "labels" not in example_batch:
            # Rename label to labels to make it easier to pass to model forward
            features["labels"] = example_batch["label"]

        return features

    def _apply_dataset_fraction(
        self, dataset: DatasetDict, fraction: float, seed: int
    ) -> DatasetDict:
        """Apply dataset fraction to each split in the DatasetDict."""
        np.random.seed(seed)
        reduced_dataset = {}

        for split, data in dataset.items():
            # Sample fraction of the data
            total_samples = int(len(data) * fraction)
            indices = np.random.choice(len(data), total_samples, replace=False)
            indices = list(indices)
            sampled_data = data.select(indices)
            reduced_dataset[split] = sampled_data

        return DatasetDict(reduced_dataset)

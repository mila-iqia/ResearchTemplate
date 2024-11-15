"""Example: fine-tuning a language model (GPT, GPT-2, CTRL, OPT, etc.) on a text dataset.

Large chunks of the code here are taken from [this example script in the transformers GitHub repository](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py).

If you haven't already, you should definitely check out [this walkthrough of that script from the HuggingFace docs.](https://huggingface.co/docs/transformers/en/tasks/language_modeling)
"""

import dataclasses
import hashlib
import itertools
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

import datasets
import datasets.distributed
import hydra_zen
import torch
import torch.distributed
from datasets import Dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
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
from project.utils.typing_utils import NestedMapping

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
class NetworkConfig:
    """Configuration options related to the choice of network.

    When instantiated by Hydra, this calls the `target` function passed to the decorator. In this
    case, this creates pulls the pretrained network weights from the HuggingFace model hub.
    """

    __doc__ = """Configuration options related to the choice of network.

When instantiated by Hydra, this calls the `target` function passed to the decorator. In this
case, this creates pulls the pretrained network weights from the HuggingFace model hub.
"""

    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    torch_dtype: torch.dtype | None = None
    attn_implementation: str | None = None
    # cache_dir: Path | None = None
    # force_download: bool | None = None
    # local_files_only: bool | None = None
    # proxies: dict[str, str] | None = None
    # revision: str | None = None
    # subfolder: str | None = None
    # use_auth_token: bool | None = None
    # token: str | bool | None = None


# BUG: Hydra-zen includes the doc of the target, so doctest complains here.
NetworkConfig.__doc__ = """Configuration options related to the choice of network.

When instantiated by Hydra, this calls the `target` function passed to the decorator. In this
case, this creates pulls the pretrained network weights from the HuggingFace model hub.
"""


@hydra_zen.hydrated_dataclass(
    target=AutoTokenizer.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TokenizerConfig:
    """Configuration options for the tokenizer."""

    pretrained_model_name_or_path: str
    cache_dir: Path | None = None  # use standard cache by default.
    force_download: bool = False
    local_files_only: bool = False
    token: str | bool | None = None
    revision: str = "main"
    use_fast: bool = True
    config: PretrainedConfig | None = None
    # proxies: dict[str, str] = dataclasses.field(default_factory=dict, hash=False)
    subfolder: str = ""
    tokenizer_type: str | None = None
    trust_remote_code: bool = False


# BUG: Hydra-zen includes the doc of the target, so doctest complains here.
TokenizerConfig.__doc__ = """Configuration options for the tokenizer."""


@dataclass(frozen=True, unsafe_hash=True)
class DatasetConfig:
    """Configuration options related to the dataset preparation."""

    dataset_path: str
    """Name of the dataset "family"?

    For example, to load "wikitext/wikitext-103-v1", this would be "wikitext".
    """

    dataset_name: str | None = None
    """Name of the specific dataset?

    For example, to load "wikitext/wikitext-103-v1", this would be "wikitext-103-v1".
    """

    # Don't include those fields when computign the 'id' of the config, which we use to determine
    # if we've already prepared the dataset or not.
    per_device_eval_batch_size: int = dataclasses.field(
        default=8, metadata={"include_in_id": False}
    )
    per_device_train_batch_size: int = dataclasses.field(
        default=8, metadata={"include_in_id": False}
    )

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


def prepare_datasets(
    dataset_config: DatasetConfig, tokenizer_config: TokenizerConfig
) -> DatasetDict:
    # todo: an improvement could be to cache each portion, so that if we just change the block
    # size, we don't have to re-tokenize the dataset for example.
    raw_datasets = load_raw_datasets(dataset_config)
    tokenizer = load_tokenizer(tokenizer_config)
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, dataset_config)
    lm_datasets = group_text_into_blocks(tokenized_datasets, tokenizer, dataset_config)
    return lm_datasets


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
        desc="Tokenizing the dataset",
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
        desc=f"Grouping tokens into chunks of size {block_size}",
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
    """Example of a lightning module used to fine-tune a huggingface model."""

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

        # NOTE: have to do this because Lightning doesn't do it automatically for dataclasses...
        self.save_hyperparameters(
            dict(
                network_config=dataclasses.asdict(network_config),
                tokenizer_config=dataclasses.asdict(tokenizer_config),
                dataset_config=dataclasses.asdict(dataset_config),
                learning_rate=learning_rate,
                adam_epsilon=adam_epsilon,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                init_seed=init_seed,
            )
        )

        # We will prepare the dataset only on the first task of the first node node for multi-node
        # jobs.
        # TODO: there is probably a way to do distributed preprocessing (tokenization/grouping/...)
        # perhaps we could look into this:
        # https://huggingface.co/docs/datasets/v3.1.0/en/use_with_pytorch#distributed
        self.prepare_data_per_node = True  # Execute `prepare_data` on each node.
        self.data_configs_id = (
            f"{get_hash_of(self.dataset_config)[:8]}_{get_hash_of(self.tokenizer_config)[:8]}"
        )
        logger.info(f"Unique id for our dataset / tokenizer configs: {self.data_configs_id}")

        self.scratch_prepared_dataset_dir: Path | None = None
        if SCRATCH is not None:
            # TODO: Should we base ourselves on the HF environment variables instead of hard-coding
            # $SCRATCH/data/...?
            self.scratch_prepared_dataset_dir = (
                SCRATCH / "data" / "prepared_dataset" / self.data_configs_id
            )
            self.scratch_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        fast_data_dir = (SLURM_TMPDIR or Path.cwd()) / "data" / "prepared_dataset"
        self.fast_prepared_dataset_dir = fast_data_dir / self.data_configs_id
        self.fast_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.train_dataset: Dataset | None = None
        self.valid_dataset: Dataset | None = None
        self.network: AutoModelForCausalLM | None = None

    def configure_model(self) -> None:
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        # Initialize the weights on the GPU if we have one, so we don't
        # request lots of RAM just to load up the model weights and then not use it.
        if self.network is not None:
            return
        logger.info(f"Rank {self.local_rank}: {self.device=}")
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)

    def prepare_data(self):
        # This gets called on every node in a distrituted training setup.
        # See the Lightning docs for this method for more information.
        #
        # If we've already prepared the dataset on this node, we can just load it.
        # If we're on a SLURM cluster and we've already prepared it in $SCRATCH, then copy it to
        # the local fast directory.
        # Otherwise do the tokenization and grouping and save it to the local fast directory, then
        # copy it to the $SCRATCH directory for future use.
        if _try_to_load_prepared_dataset_from(self.fast_prepared_dataset_dir):
            logger.info(
                f"Dataset is already prepared on this node at {self.fast_prepared_dataset_dir}"
            )
            return
        logger.debug("Dataset hasn't been prepared on this node yet.")

        if not self.scratch_prepared_dataset_dir:
            # Let's assume that you're using SLURM for multi-node jobs for now.
            # SCRATCH isn't set --> not on a SLURM cluster.
            assert self.trainer.num_nodes == 1
            logger.info(f"Preparing dataset at {self.fast_prepared_dataset_dir}.")
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            return

        if _try_to_load_prepared_dataset_from(self.scratch_prepared_dataset_dir):
            logger.info(
                f"Dataset is already prepared on the shared filesystem at "
                f"{self.scratch_prepared_dataset_dir}"
            )
            copy_dataset_files(self.scratch_prepared_dataset_dir, self.fast_prepared_dataset_dir)
            return

        logger.debug("Dataset has not yet been prepared with this config yet.")

        if self.trainer.num_nodes == 1:
            logger.debug("Single-node training. Preparing the dataset.")
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            logger.info(f"Saved processed dataset to {self.fast_prepared_dataset_dir}")
            copy_dataset_files(self.fast_prepared_dataset_dir, self.scratch_prepared_dataset_dir)
            return

        # NOTE: There might be a way to distribute the preprocessing across nodes, I'm not sure.
        # todo: Would be even better to add an `srun` step before this with `ntasks_per_node=1` to
        # speed up the preprocessing!
        _barrier_name = "prepare_data"
        if self.global_rank == 0:
            logger.info(
                f"Task {self.global_rank}: Preparing the dataset in $SLURM_TMPDIR and copying it to $SCRATCH."
            )
            # TODO: This might cause some timeouts if the dataset preprocessing takes a while to do, no?
            # TODO:
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            logger.info(f"Saved processed dataset to {self.fast_prepared_dataset_dir}")
            copy_dataset_files(self.fast_prepared_dataset_dir, self.scratch_prepared_dataset_dir)
            logger.info(f"Task {self.global_rank}: Done preparing the dataset.")
            # wait (i.e. join the other tasks that are already waiting)
            self.trainer.strategy.barrier(_barrier_name)
        else:
            logger.info(
                f"Task {self.global_rank}: Waiting for the first task on the first node to finish preparing the dataset."
            )
            # Wait for the first task to get to the barrier (i.e. wait for the first task to finish
            # preprocessing the dataset).
            self.trainer.strategy.barrier(_barrier_name)

            assert self.scratch_prepared_dataset_dir.exists()
            logger.info(
                f"Copying the dataset files prepared by the first node at {self.scratch_prepared_dataset_dir}"
            )
            copy_dataset_files(self.scratch_prepared_dataset_dir, self.fast_prepared_dataset_dir)

        logger.info(f"Done preparing the datasets at {self.fast_prepared_dataset_dir}.")

    def setup(self, stage: str):
        """Hook from Lightning that is called at the start of training, validation and testing.

        TODO: Later perhaps we could do the preprocessing in a distributed manner like this:
        https://discuss.huggingface.co/t/how-to-save-datasets-as-distributed-with-save-to-disk/25674/2
        """
        # https://huggingface.co/docs/datasets/v3.1.0/en/use_with_pytorch#distributed
        # Load the tokenizer (again).
        self.tokenizer = load_tokenizer(self.tokenizer_config)
        lm_datasets = datasets.load_from_disk(self.fast_prepared_dataset_dir)

        # This is done here again because in distributed training jobs, `prepare_data` is only
        # called in the first task on each node, while `setup` is called in every task.
        logger.info(f"Loading processed dataset from {self.fast_prepared_dataset_dir}")
        assert isinstance(lm_datasets, DatasetDict)
        self.train_dataset = lm_datasets["train"]
        self.valid_dataset = lm_datasets["validation"]

        # todo: Should we be using `datasets.distributed.split_dataset_by_node` here? Or do we let
        # PyTorch-Lightning setup the distributed sampler for us?
        # self.train_dataset = datasets.distributed.split_dataset_by_node(
        #     self.train_dataset, rank=self.global_rank, world_size=self.trainer.world_size
        # )
        # self.valid_dataset = datasets.distributed.split_dataset_by_node(
        #     self.valid_dataset, rank=self.global_rank, world_size=self.trainer.world_size
        # )

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.dataset_config.preprocessing_num_workers,
            batch_size=self.dataset_config.per_device_train_batch_size,
        )

    def val_dataloader(self):
        assert self.valid_dataset is not None

        return DataLoader(
            self.valid_dataset,
            collate_fn=default_data_collator,
            num_workers=self.dataset_config.preprocessing_num_workers,
            batch_size=self.dataset_config.per_device_eval_batch_size,
        )

    def forward(self, **inputs: torch.Tensor) -> BaseModelOutput:
        assert self.network is not None
        return self.network(**inputs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs: CausalLMOutput | SequenceClassifierOutput = self(**batch)
        loss = outputs.loss
        assert loss is not None
        # todo: log more stuff!
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        outputs: CausalLMOutput | SequenceClassifierOutput = self(**batch)
        loss = outputs.loss
        assert loss is not None
        # todo: log more stuff!
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # Not sure if necessary, but trying to follow this recommendation for when using FSDP:
        # https://github.com/ashleve/lightning-hydra-template/pull/604
        model = self.trainer.model or self
        assert model is not None

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd_param in n for nd_param in no_decay)
                ],
                "weight_decay": self.weight_decay,
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def copy_dataset_files(src: Path, dest: Path):
    logger.info(f"Copying dataset from {src} --> {dest}")
    shutil.copytree(src, dest)


def get_hash_of(config_dataclass) -> str:
    # IDEA: don't include fields if they have `hash=False` in the "hash".
    vals = dataclasses.asdict(config_dataclass)
    for field in dataclasses.fields(config_dataclass):
        if not _include_field_in_id(field):
            logger.debug(f"Ignoring field {field.name} when computing the ID.")
            vals.pop(field.name)

    flattened_vals = dict(sorted(flatten_dict(vals).items()))
    vals_string = ",".join(f"{k}:{v}" for k, v in flattened_vals.items())
    return hashlib.md5(vals_string.encode()).hexdigest()


V = TypeVar("V")


def flatten_dict(d: NestedMapping[str, V]) -> dict[str, V]:
    result = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result.update({f"{k}.{subk}": subv for subk, subv in flatten_dict(v).items()})
        else:
            result[k] = v
    return result


P = ParamSpec("P")


def _try_to_load_prepared_dataset_from(
    dataset_path: Path,
    _load_from_disk_fn: Callable[Concatenate[Path, P], Dataset | DatasetDict] = load_from_disk,
    *_load_from_disk_args: P.args,
    **_load_from_disk_kwargs: P.kwargs,
) -> DatasetDict | None:
    try:
        datasets = _load_from_disk_fn(
            dataset_path, *_load_from_disk_args, **_load_from_disk_kwargs
        )
    except FileNotFoundError as exc:
        logger.debug(f"Unable to load the prepared dataset from {dataset_path}: {exc}")
        return None
    else:
        logger.debug(f"Dataset is already prepared at {dataset_path}")
        assert isinstance(datasets, DatasetDict)
        return datasets


def _include_field_in_id(field: dataclasses.Field) -> bool:
    return field.metadata.get("include_in_id", True)

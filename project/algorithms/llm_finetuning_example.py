import functools
from pathlib import Path

import datasets
import torch
from lightning import LightningModule
from torch.optim.adamw import AdamW
from transformers import (
    AutoConfig,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel


def pretrained_network(model_name_or_path: str | Path, **kwargs) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    return AutoModel.from_pretrained(model_name_or_path, config=config)


class LLMFinetuningExample(LightningModule):
    """Example of a lightning module used to train a huggingface model."""

    def __init__(
        self,
        network: PreTrainedModel | functools.partial[PreTrainedModel],
        dataset_path: str,
        dataset_name: str | None = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
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

        with torch.random.fork_rng():
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            if isinstance(network, functools.partial):
                network = network()
            self.network = network

        # Small fix for the `device` property in LightningModule, which is CPU by default.
        self._device = next((p.device for p in self.parameters()), torch.device("cpu"))
        self.save_hyperparameters(ignore=["network", "datamodule"])

    def prepare_data(self):
        raw_datasets = datasets.load_dataset(self.dataset_path, self.dataset_name)

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = datasets.load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=f"train[:{self.validation_split_percentage}%]",
            )
            raw_datasets["train"] = datasets.load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=f"train[{self.validation_split_percentage}%:]",
            )

    def setup(self, stage: str):
        self.datasets = datasets.load_dataset(self.dataset_path, self.dataset_name)

    def train_dataloader(self): ...

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

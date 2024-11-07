import functools
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import evaluate
import torch
from lightning import LightningModule
from torch.optim.adamw import AdamW
from transformers import (
    AutoConfig,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.auto.modeling_auto import AutoModel

from project.datamodules.text.hf_text import HFDataModule


def pretrained_network(model_name_or_path: str | Path, **kwargs) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    return AutoModel.from_pretrained(model_name_or_path, config=config)


class HFExample(LightningModule):
    """Example of a lightning module used to train a huggingface model."""

    def __init__(
        self,
        datamodule: HFDataModule,
        network: PreTrainedModel | Callable[..., PreTrainedModel],
        hf_metric_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        init_seed: int = 42,
    ):
        super().__init__()

        self.num_labels = getattr(datamodule, "num_classes", None)
        self.task_name = datamodule.task_name
        self.init_seed = init_seed
        self.hf_metric_name = hf_metric_name
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

        self.metric = evaluate.load(
            self.hf_metric_name,
            self.task_name,
            # todo: replace with hydra job id perhaps?
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )

        # Small fix for the `device` property in LightningModule, which is CPU by default.
        self._device = next((p.device for p in self.parameters()), torch.device("cpu"))
        self.save_hyperparameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        return self.network(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels
        )

    def model_step(self, batch: dict[str, torch.Tensor]):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(input_ids, token_type_ids, attention_mask, labels)
        loss = outputs.loss
        logits = outputs.logits

        if self.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        else:
            preds = logits.squeeze()

        return loss, preds, labels

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        val_loss, preds, labels = self.model_step(batch)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val/loss": val_loss, "preds": preds, "labels": labels}

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
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

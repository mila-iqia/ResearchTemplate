from datetime import datetime

import evaluate
import hydra_zen
import torch
from lightning import LightningModule
from torch.optim.adamw import AdamW
from transformers import (
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput

from project.datamodules.text.text_classification import TextClassificationDataModule
from project.utils.typing_utils import HydraConfigFor


class TextClassifier(LightningModule):
    """Example of a lightning module used to train a huggingface model for text classification."""

    def __init__(
        self,
        datamodule: TextClassificationDataModule,
        network: HydraConfigFor[PreTrainedModel],
        hf_metric_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        init_seed: int = 42,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.num_labels = datamodule.num_classes
        self.task_name = datamodule.task_name
        self.init_seed = init_seed
        self.hf_metric_name = hf_metric_name
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self.metric = evaluate.load(
            self.hf_metric_name,
            self.task_name,
            # todo: replace with hydra job id perhaps?
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )

        self.save_hyperparameters(ignore=["datamodule"])

    def configure_model(self) -> None:
        with torch.random.fork_rng(devices=[self.device]):
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)

        return super().configure_model()

    def forward(self, inputs: dict[str, torch.Tensor]) -> BaseModelOutput:
        return self.network(**inputs)

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str):
        outputs: CausalLMOutput | SequenceClassifierOutput = self(batch)
        loss = outputs.loss
        assert isinstance(loss, torch.Tensor), loss
        # todo: log the output of the metric.
        self.log(f"{stage}/loss", loss, prog_bar=True)
        if isinstance(outputs, SequenceClassifierOutput):
            metric_value = self.metric.compute(
                # logits=outputs.logits,
                predictions=outputs.logits.argmax(-1),
                references=batch["labels"],
            )
            assert isinstance(metric_value, dict)
            for k, v in metric_value.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    prog_bar=True,
                )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ):
        return self.shared_step(batch, batch_idx, "val")

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
        return [optimizer], [scheduler]

import warnings
from logging import getLogger as get_logger
from typing import Required

import torch
import torchmetrics
from lightning import LightningModule, Trainer
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from project.algorithms.bases.algorithm import Algorithm, BatchType
from project.algorithms.bases.image_classification import StepOutputDict
from project.algorithms.callbacks.callback import Callback
from project.utils.types import PhaseStr, StageStr
from project.utils.types.protocols import ClassificationDataModule

logger = get_logger(__name__)


class ClassificationOutputs(StepOutputDict):
    """The dictionary format that is minimally required to be returned from
    `training/val/test_step` for classification algorithms."""

    logits: Required[Tensor]
    """The un-normalized logits."""

    y: Required[Tensor]
    """The class labels."""


class ClassificationMetricsCallback(Callback[BatchType, ClassificationOutputs]):
    def __init__(self) -> None:
        super().__init__()
        self.disabled = False

    @classmethod
    def attach_to(cls, algorithm: Algorithm, num_classes: int):
        callback = cls()
        callback.add_metrics_to(algorithm, num_classes=num_classes)
        return callback

    def add_metrics_to(self, pl_module: LightningModule, num_classes: int) -> None:
        # IDEA: Could use a dict of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: dist[str, Metrics]
        # NOTE: Need to have one per phase! Not 100% sure that I'm not forgetting a phase here.

        # Slightly ugly. Need to set the metrics on the pl module for things to be logged / synced
        # easily.
        metrics_to_add = {
            "train_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "val_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "test_accuracy": MulticlassAccuracy(num_classes=num_classes),
            "train_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
            "val_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
            "test_top5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5),
        }
        if all(
            hasattr(pl_module, name) and isinstance(getattr(pl_module, name), type(metric))
            for name, metric in metrics_to_add.items()
        ):
            logger.info("Not adding metrics to the pl module because they are already present.")
            return

        for metric_name, metric in metrics_to_add.items():
            self._set_metric(pl_module, metric_name, metric)

    # todo: change these two if we end up putting metrics in a ModuleDict.
    @staticmethod
    def _set_metric(pl_module: LightningModule, name: str, metric: torchmetrics.Metric):
        if hasattr(pl_module, name):
            raise RuntimeError(f"The pl module already has an attribute with the name {name}.")
        logger.info(f"Setting a new metric on the pl module at attribute {name}.")
        setattr(pl_module, name, metric)

    @staticmethod
    def _get_metric(pl_module: LightningModule, name: str):
        return getattr(pl_module, name)

    def setup(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, ClassificationOutputs],
        stage: StageStr,
    ) -> None:
        if self.disabled:
            return
        datamodule = pl_module.datamodule
        if not isinstance(datamodule, ClassificationDataModule):
            warnings.warn(
                RuntimeWarning(
                    f"Disabling the {type(self).__name__} callback because it only works with "
                    f"classification datamodules, but {pl_module.datamodule=} isn't a "
                    f"{ClassificationDataModule.__name__}."
                )
            )
            self.disabled = True
            return

        num_classes = datamodule.num_classes
        self.add_metrics_to(pl_module, num_classes=num_classes)

    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, ClassificationOutputs],
        outputs: ClassificationOutputs,
        batch: BatchType,
        batch_index: int,
        phase: PhaseStr,
        dataloader_idx: int | None = None,
    ):
        if self.disabled:
            return
        step_output = outputs
        required_entries = ClassificationOutputs.__required_keys__
        if not isinstance(outputs, dict):
            warnings.warn(
                RuntimeWarning(
                    f"Expected the {phase} step method to output a dictionary with at least the "
                    f"{required_entries} keys, but got an output of type {type(step_output)} instead!\n"
                    f"Disabling the {type(self).__name__} callback."
                )
            )
            self.disabled = True
            return
        if not all(k in step_output for k in required_entries):
            warnings.warn(
                RuntimeWarning(
                    f"Expected all the following keys to be in the output of the {phase} step "
                    f"method: {required_entries}. Disabling the {type(self).__name__} callback."
                )
            )
            self.disabled = True
            return

        logits = step_output["logits"]
        y = step_output["y"]

        probs = torch.softmax(logits, -1)

        accuracy = self._get_metric(pl_module, f"{phase}_accuracy")
        top5_accuracy = self._get_metric(pl_module, f"{phase}_top5_accuracy")
        assert isinstance(accuracy, MulticlassAccuracy)
        assert isinstance(top5_accuracy, MulticlassAccuracy)

        # TODO: It's a bit confusing, not sure if this is the right way to use this:
        accuracy(probs, y)
        top5_accuracy(probs, y)
        prog_bar = phase == "train"

        pl_module.log(f"{phase}/accuracy", accuracy, prog_bar=prog_bar, sync_dist=True)
        pl_module.log(f"{phase}/top5_accuracy", top5_accuracy, prog_bar=prog_bar, sync_dist=True)

        if "cross_entropy" not in step_output:
            # Add the cross entropy loss as a metric.
            with torch.no_grad():
                ce_loss = torch.nn.functional.cross_entropy(logits.detach(), y, reduction="mean")
            pl_module.log(f"{phase}/cross_entropy", ce_loss, prog_bar=prog_bar, sync_dist=True)

        loss: Tensor | float | None = step_output.get("loss", None)
        if loss is not None:
            # note: Perhaps we should be careful not to overwrite the logged value if its already been logged?
            pl_module.log(
                f"{phase}/loss", torch.as_tensor(loss).mean(), prog_bar=prog_bar, sync_dist=True
            )

        # This part isn't necessary here: Average out the losses properly.
        # fused_output = step_output.copy()
        # if isinstance(loss, Tensor) and loss.shape:
        #     # Replace the loss with its mean. This is useful when automatic
        #     # optimization is enabled, for example in the baseline (backprop), where each replica
        #     # returns the un-reduced cross-entropy loss. Here we need to reduce it to a scalar.
        #     fused_output["loss"] = loss.mean()

        # return fused_output

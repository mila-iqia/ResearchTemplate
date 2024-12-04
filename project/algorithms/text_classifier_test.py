from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning
import pytest
import torch
from lightning import LightningModule
from tensor_regression import TensorRegressionFixture
from torch import Tensor
from transformers import PreTrainedModel
from typing_extensions import override

from project.algorithms.text_classifier import TextClassifier
from project.datamodules.text.text_classification import TextClassificationDataModule
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.testutils import run_for_all_configs_of_type, total_vram_gb

from .testsuites.lightning_module_tests import GetStuffFromFirstTrainingStep, LightningModuleTests


class RecordTrainingLossCb(lightning.Callback):
    def __init__(self):
        self.losses: list[Tensor] = []

    @override
    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ):
        assert isinstance(outputs, dict) and isinstance(loss := outputs.get("loss"), Tensor)
        self.losses.append(loss.detach())


@pytest.mark.skipif(total_vram_gb() < 16, reason="Not enough VRAM to run this test.")
@run_for_all_configs_of_type("algorithm", TextClassifier)
@run_for_all_configs_of_type("datamodule", TextClassificationDataModule)
@run_for_all_configs_of_type("algorithm/network", PreTrainedModel)
class TestTextClassifier(LightningModuleTests[TextClassifier]):
    """Tests for the HF example."""

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None,
        reason="Weird reproducibility issue with HuggingFace model/dataset?",
        raises=AssertionError,
    )
    def test_backward_pass_is_reproducible(  # type: ignore
        self,
        training_step_content: tuple[
            TextClassifier, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        return super().test_backward_pass_is_reproducible(
            training_step_content=training_step_content,
            tensor_regression=tensor_regression,
            accelerator=accelerator,
        )

    @pytest.mark.skip(reason="TODO: Seems to be causing issues due to DDP?")
    @pytest.mark.slow
    def test_overfit_batch(
        self,
        algorithm: TextClassifier,
        datamodule: TextClassificationDataModule,
        tmp_path: Path,
        num_steps: int = 3,
    ):
        """Test that the loss decreases on a single batch."""
        get_loss_cb = RecordTrainingLossCb()
        trainer = lightning.Trainer(
            accelerator="auto",
            strategy="auto",
            callbacks=[get_loss_cb],
            devices=[0] if torch.cuda.is_available() else "auto",
            enable_checkpointing=False,
            deterministic=True,
            default_root_dir=tmp_path,
            overfit_batches=1,
            limit_train_batches=1,
            max_epochs=num_steps,
        )
        trainer.fit(algorithm, datamodule)
        losses_at_each_epoch: list[Tensor] = get_loss_cb.losses

        assert (
            len(losses_at_each_epoch) == num_steps
        ), f"Expected {num_steps} losses, got {len(losses_at_each_epoch)}"

        assert losses_at_each_epoch[0] > losses_at_each_epoch[-1], (
            f"Loss did not decrease on overfit: final loss= {losses_at_each_epoch[-1]},"
            f"initial loss={losses_at_each_epoch[0]}"
        )

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

from project.algorithms.hf_example import HFExample
from project.datamodules.text.hf_text import HFDataModule
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.algorithm_tests import LearningAlgorithmTests


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


def total_vram_gb() -> float:
    """Returns the total VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


# TODO: There's a failing test here only on SLURM?


@pytest.mark.skipif(total_vram_gb() < 16, reason="Not enough VRAM to run this test.")
@run_for_all_configs_of_type("algorithm", HFExample)
@run_for_all_configs_of_type("datamodule", HFDataModule)
@run_for_all_configs_of_type("algorithm/network", PreTrainedModel)
class TestHFExample(LearningAlgorithmTests[HFExample]):
    """Tests for the HF example."""

    @pytest.mark.xfail(
        SLURM_JOB_ID is not None,
        reason="Weird reproducibility issue with HuggingFace model/dataset on the cluster?",
        raises=AssertionError,
    )
    def test_backward_pass_is_reproducible(  # type: ignore
        self,
        datamodule: HFDataModule,
        algorithm: HFExample,
        seed: int,
        accelerator: str,
        devices: int | list[int],
        tensor_regression: TensorRegressionFixture,
        tmp_path: Path,
    ):
        return super().test_backward_pass_is_reproducible(
            datamodule=datamodule,
            algorithm=algorithm,
            seed=seed,
            accelerator=accelerator,
            devices=devices,
            tensor_regression=tensor_regression,
            tmp_path=tmp_path,
        )

    @pytest.mark.skip(reason="TODO: Seems to be causing issues due to DDP?")
    @pytest.mark.slow
    def test_overfit_batch(
        self,
        algorithm: HFExample,
        datamodule: HFDataModule,
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

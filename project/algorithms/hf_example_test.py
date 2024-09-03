import lightning
import pytest
from torch import Tensor
from transformers import PreTrainedModel

from project.algorithms.hf_example import HFExample
from project.datamodules.text.hf_text import HFDataModule
from project.utils.testutils import run_for_all_configs_of_type

from .testsuites.algorithm_tests import LearningAlgorithmTests


class RecordTrainingLossCb(lightning.Callback):
    def __init__(self):
        self.losses = []

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        self.losses.append(outputs["loss"].detach())


@run_for_all_configs_of_type("algorithm", HFExample)
@run_for_all_configs_of_type("datamodule", HFDataModule)
@run_for_all_configs_of_type("network", PreTrainedModel)
class TestHFExample(LearningAlgorithmTests[HFExample]):
    """Tests for the HF example."""

    @pytest.fixture(scope="session")
    def forward_pass_input(self, training_batch: dict[str, Tensor]):
        assert isinstance(training_batch, dict)
        return training_batch

    @pytest.mark.slow
    def test_overfit_batch(
        self,
        algorithm: HFExample,
        datamodule: HFDataModule,
        accelerator: str,
        devices: int | list[int],
        training_batch: dict[str, Tensor],
        num_steps: int = 3,
    ):
        """Test that the loss decreases on a single batch."""
        get_loss_cb = RecordTrainingLossCb()
        trainer = lightning.Trainer(
            accelerator=accelerator,
            callbacks=[get_loss_cb],
            devices=devices,
            enable_checkpointing=False,
            deterministic=True,
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

import time

from lightning import Trainer
from torch import Tensor, nn

from project.algorithms.bases.algorithm import Algorithm, BatchType
from project.algorithms.callbacks.callback import Callback
from project.utils.types import PhaseStr, StepOutputDict, is_sequence_of


class MeasureSamplesPerSecondCallback(Callback[BatchType, StepOutputDict]):
    def __init__(self):
        super().__init__()
        self.last_step_times: dict[PhaseStr, float] = {}

    def on_shared_epoch_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputDict, nn.Module],
        phase: PhaseStr,
    ) -> None:
        self.last_step_times.pop(phase, None)

    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputDict, nn.Module],
        outputs: StepOutputDict,
        batch: BatchType,
        batch_idx: int,
        phase: PhaseStr,
        dataloader_idx: int | None = None,
    ):
        super().on_shared_batch_end(
            trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            phase=phase,
            dataloader_idx=dataloader_idx,
        )
        now = time.perf_counter()
        if phase in self.last_step_times:
            elapsed = now - self.last_step_times[phase]
            if is_sequence_of(batch, Tensor):
                batch_size = batch[0].shape[0]
                pl_module.log(f"{phase}/samples_per_second", batch_size / elapsed, prog_bar=True)
            # todo: support other kinds of batches
        self.last_step_times[phase] = now

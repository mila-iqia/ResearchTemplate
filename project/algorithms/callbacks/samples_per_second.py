import time
from typing import Literal, override

from lightning import LightningModule, Trainer
from torch import Tensor
from torch.optim import Optimizer

from project.algorithms.callbacks.callback import BatchType, Callback, StepOutputType
from project.utils.types import is_sequence_of


class MeasureSamplesPerSecondCallback(Callback[BatchType, StepOutputType]):
    def __init__(self):
        super().__init__()
        self.last_step_times: dict[Literal["train", "val", "test"], float] = {}
        self.last_update_time: dict[int, float | None] = {}
        self.num_optimizers: int | None = None

    @override
    def on_shared_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        phase: Literal["train", "val", "test"],
    ) -> None:
        self.last_update_time.clear()
        self.last_step_times.pop(phase, None)
        if self.num_optimizers is None:
            optimizer_or_optimizers = pl_module.optimizers()
            if not isinstance(optimizer_or_optimizers, list):
                self.num_optimizers = 1
            else:
                self.num_optimizers = len(optimizer_or_optimizers)

    @override
    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        phase: Literal["train", "val", "test"],
        dataloader_idx: int | None = None,
    ):
        super().on_shared_batch_end(
            trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            phase=phase,
            dataloader_idx=dataloader_idx,
        )
        now = time.perf_counter()
        if phase in self.last_step_times:
            elapsed = now - self.last_step_times[phase]
            if is_sequence_of(batch, Tensor):
                batch_size = batch[0].shape[0]
                pl_module.log(
                    f"{phase}/samples_per_second",
                    batch_size / elapsed,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            # todo: support other kinds of batches
        self.last_step_times[phase] = now

    @override
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        opt_idx: int = 0,
    ) -> None:
        if opt_idx not in self.last_update_time or self.last_update_time[opt_idx] is None:
            self.last_update_time[opt_idx] = time.perf_counter()
            return
        last_update_time = self.last_update_time[opt_idx]
        assert last_update_time is not None
        now = time.perf_counter()
        elapsed = now - last_update_time
        updates_per_second = 1 / elapsed
        if self.num_optimizers == 1:
            key = "ups"
        else:
            key = f"optimizer_{opt_idx}/ups"
        pl_module.log(
            key,
            updates_per_second,
            prog_bar=False,
            on_step=True,
        )

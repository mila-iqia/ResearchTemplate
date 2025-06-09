import time
from typing import Generic, Literal

import lightning
import optree
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeVar, override

from project.utils.typing_utils import NestedMapping, is_sequence_of

BatchType = TypeVar(
    "BatchType",
    bound=torch.Tensor | tuple[torch.Tensor, ...] | NestedMapping[str, torch.Tensor],
    contravariant=True,
)


class MeasureSamplesPerSecondCallback(lightning.Callback, Generic[BatchType]):
    """Callback that measures the number of samples processed per second during train/val/test."""

    def __init__(self, num_optimizers: int | None = None):
        super().__init__()
        self.last_step_times: dict[Literal["train", "val", "test"], float] = {}
        self.last_update_time: dict[int, float | None] = {}
        self.num_optimizers: int | None = num_optimizers

    @override
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_shared_epoch_start(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_shared_epoch_start(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_shared_epoch_start(trainer, pl_module, phase="test")

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
    def on_train_batch_end(
        self,
        trainer: Trainer | None,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: BatchType,
        batch_idx: int | None,
    ) -> None:
        self.on_shared_batch_end(pl_module=pl_module, batch=batch, phase="train")

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer | None,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: BatchType,
        batch_idx: int | None,
        dataloader_idx: int | None = 0,
    ) -> None:
        self.on_shared_batch_end(pl_module=pl_module, batch=batch, phase="val")

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer | None,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: BatchType,
        batch_idx: int | None = None,
        dataloader_idx: int | None = 0,
    ) -> None:
        self.on_shared_batch_end(pl_module=pl_module, batch=batch, phase="test")

    def on_shared_batch_end(
        self, pl_module: LightningModule, batch: BatchType, phase: Literal["train", "val", "test"]
    ):
        # Note: Not using use cuda events here, since we just want a rough throughput estimate,
        # and we assume that there's at least one synchronize call at each step.
        now = time.perf_counter()
        if phase in self.last_step_times:
            elapsed = now - self.last_step_times[phase]
            batch_size = self.get_num_samples(batch)
            pl_module.log(
                f"{phase}/samples_per_second",
                batch_size / elapsed,
                # module=pl_module,
                # trainer=trainer,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            # todo: support other kinds of batches
        self.last_step_times[phase] = now

    def get_num_samples(self, batch: BatchType) -> int:
        if isinstance(batch, Tensor):
            return batch.shape[0]
        if is_sequence_of(batch, Tensor):
            return batch[0].shape[0]
        if isinstance(batch, dict):
            return next(
                v.shape[0]
                for v in optree.tree_leaves(batch)  # type: ignore
                if isinstance(v, torch.Tensor) and v.ndim > 1
            )
        raise NotImplementedError(
            f"Don't know how many 'samples' there are in batch of type {type(batch)}"
        )

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
            # module=pl_module,
            # trainer=trainer,
            prog_bar=False,
            on_step=True,
        )

from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
from typing import override

from lightning import Trainer
from lightning import pytorch as pl
from typing_extensions import Generic  # noqa

from project.algorithms.bases.algorithm import Algorithm, BatchType, StepOutputType
from project.utils.types import PhaseStr, StageStr
from project.utils.utils import get_log_dir

logger = get_logger(__name__)


class Callback(pl.Callback, Generic[BatchType, StepOutputType]):
    """Adds a bit of typing info and shared functions to the PyTorch Lightning Callback class."""

    def __init__(self) -> None:
        super().__init__()
        self.log_dir: Path | None = None

    @override
    def setup(
        self, trainer: pl.Trainer, pl_module: Algorithm[BatchType, StepOutputType], stage: StageStr
    ) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        self.log_dir = get_log_dir(trainer=trainer)

    def on_shared_batch_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        batch: BatchType,
        batch_index: int,
        phase: PhaseStr,
        dataloader_idx: int | None = None,
    ): ...

    def on_shared_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        phase: PhaseStr,
        dataloader_idx: int | None = None,
    ): ...

    def on_shared_epoch_start(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType], phase: PhaseStr
    ) -> None: ...

    def on_shared_epoch_end(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType], phase: PhaseStr
    ) -> None: ...

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
    ) -> None:
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_index=batch_index,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            phase="train",
        )

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int,
    ) -> None:
        super().on_validation_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_index=batch_index,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            dataloader_idx=dataloader_idx,
            phase="val",
        )

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int,
    ) -> None:
        super().on_test_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_index=batch_index,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            dataloader_idx=dataloader_idx,
            phase="test",
        )

    @override
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        batch: BatchType,
        batch_index: int,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_index)
        self.on_shared_batch_start(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            batch_index=batch_index,
            phase="train",
        )

    @override
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int,
    ) -> None:
        super().on_validation_batch_start(trainer, pl_module, batch, batch_index, dataloader_idx)
        self.on_shared_batch_start(
            trainer,
            pl_module,
            batch,
            batch_index,
            dataloader_idx=dataloader_idx,
            phase="val",
        )

    @override
    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: Algorithm[BatchType, StepOutputType],
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int,
    ) -> None:
        super().on_test_batch_start(trainer, pl_module, batch, batch_index, dataloader_idx)
        self.on_shared_batch_start(
            trainer,
            pl_module,
            batch,
            batch_index,
            dataloader_idx=dataloader_idx,
            phase="test",
        )

    @override
    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_start(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="test")

    @override
    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: Algorithm[BatchType, StepOutputType]
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="test")

from __future__ import annotations

from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, Literal

import torch
from lightning import LightningModule, Trainer
from lightning import pytorch as pl
from typing_extensions import TypeVar, override

from project.utils.typing_utils import NestedMapping
from project.utils.utils import get_log_dir

logger = get_logger(__name__)

BatchType = TypeVar(
    "BatchType",
    bound=torch.Tensor | tuple[torch.Tensor, ...] | NestedMapping[str, torch.Tensor],
    contravariant=True,
)
StepOutputType = TypeVar(
    "StepOutputType",
    bound=torch.Tensor | Mapping[str, Any] | None,
    default=dict[str, torch.Tensor],
    contravariant=True,
)


class Callback(pl.Callback, Generic[BatchType, StepOutputType]):
    """Adds a bit of typing info and shared functions to the PyTorch Lightning Callback class.

    Adds the following typing information:
    - The type of inputs that the algorithm takes
    - The type of outputs that are returned by the algorithm's `[training/validation/test]_step` methods.

    Adds the following methods:
    - `on_shared_batch_start`: called by `on_[train/validation/test]_batch_start`
    - `on_shared_batch_end`: called by `on_[train/validation/test]_batch_end`
    - `on_shared_epoch_start`: called by `on_[train/validation/test]_epoch_start`
    - `on_shared_epoch_end`: called by `on_[train/validation/test]_epoch_end`
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_dir: Path | None = None

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
        # todo: "tune" is mentioned in the docstring, is it still used?
        stage: Literal["fit", "validate", "test", "predict", "tune"],
    ) -> None:
        self.log_dir = get_log_dir(trainer=trainer)

    def on_shared_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: BatchType,
        batch_index: int,
        phase: Literal["train", "val", "test"],
        dataloader_idx: int | None = None,
    ):
        """Shared hook, called by `on_[train/validation/test]_batch_start`.

        Use this if you want to do something at the start of batches in more than one phase.
        """

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
        """Shared hook, called by `on_[train/validation/test]_batch_end`.

        Use this if you want to do something at the end of batches in more than one phase.
        """

    def on_shared_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        phase: Literal["train", "val", "test"],
    ) -> None:
        """Shared hook, called by `on_[train/validation/test]_epoch_start`.

        Use this if you want to do something at the start of epochs in more than one phase.
        """

    def on_shared_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        phase: Literal["train", "val", "test"],
    ) -> None:
        """Shared hook, called by `on_[train/validation/test]_epoch_end`.

        Use this if you want to do something at the end of epochs in more than one phase.
        """

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
    ) -> None:
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_index,
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
        pl_module: LightningModule,
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_idx=batch_index,
            dataloader_idx=dataloader_idx,
        )
        self.on_shared_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_index=batch_index,
            phase="val",
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: StepOutputType,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,  # type: ignore
            batch=batch,
            batch_idx=batch_index,
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
        pl_module: LightningModule,
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
        pl_module: LightningModule,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int = 0,
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
        pl_module: LightningModule,
        batch: BatchType,
        batch_index: int,
        dataloader_idx: int = 0,
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
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.on_shared_epoch_start(trainer, pl_module, phase="test")

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="train")

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="val")

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        self.on_shared_epoch_end(trainer, pl_module, phase="test")

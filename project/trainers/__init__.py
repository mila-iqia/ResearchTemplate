"""Trainers: actually run the training loop.

You can define custom trainers here.
"""

from lightning.pytorch.trainer.trainer import Trainer as LightningTrainer

from .jax_trainer import JaxTrainer

__all__ = [
    "JaxTrainer",
    "LightningTrainer",
]

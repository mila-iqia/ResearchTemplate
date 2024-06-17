from logging import getLogger as get_logger
from pathlib import Path

from hydra_zen import hydrated_dataclass, instantiate, store

from project.datamodules import (
    VisionDataModule,
)
from project.utils.env_vars import DATA_DIR, NETWORK_DIR, NUM_WORKERS

logger = get_logger(__name__)

torchvision_dir: Path | None = None
"""Network directory with torchvision datasets."""
if (
    NETWORK_DIR
    and (_torchvision_dir := NETWORK_DIR / "datasets/torchvision").exists()
    and _torchvision_dir.is_dir()
):
    torchvision_dir = _torchvision_dir


datamodule_store = store(group="datamodule")


@hydrated_dataclass(target=VisionDataModule, populate_full_signature=True)
class VisionDataModuleConfig:
    data_dir: str | None = str(torchvision_dir or DATA_DIR)
    val_split: int | float = 0.1  # NOTE: reduced from default of 0.2
    num_workers: int = NUM_WORKERS
    normalize: bool = True  # NOTE: Set to True by default instead of False
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = True  # NOTE: Set to True by default instead of False.
    pin_memory: bool = True  # NOTE: Set to True by default instead of False.
    drop_last: bool = False

    __call__ = instantiate


datamodule_store(VisionDataModuleConfig, name="vision")

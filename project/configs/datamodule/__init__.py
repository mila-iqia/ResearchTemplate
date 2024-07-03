from logging import getLogger as get_logger
from pathlib import Path

from hydra_zen import store
from lightning import LightningDataModule

from project.utils.env_vars import NETWORK_DIR

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


def from_datasets(
    *args, num_classes: int | None = None, dims: tuple[int, ...] | None = None, **kwargs
):
    datamodule = LightningDataModule.from_datasets(*args, **kwargs)
    if num_classes is not None:
        datamodule.num_classes = num_classes  # type: ignore
    if dims is not None:
        datamodule.dims = dims  # type: ignore
    return datamodule

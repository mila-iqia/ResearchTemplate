import importlib
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


# TODO: Make it possible to extend a structured base via yaml files as well as adding new fields
# (for example, ImagetNet32DataModule has a new constructor argument which can't be set atm in the
# config).
datamodule_store = store(group="datamodule")


# @hydrated_dataclass(target=VisionDataModule, populate_full_signature=True)
# class VisionDataModuleConfig:
#     data_dir: str | None = str(torchvision_dir or DATA_DIR)
#     val_split: int | float = 0.1  # NOTE: reduced from default of 0.2
#     num_workers: int = NUM_WORKERS
#     normalize: bool = True  # NOTE: Set to True by default instead of False
#     batch_size: int = 32
#     seed: int = 42
#     shuffle: bool = True  # NOTE: Set to True by default instead of False.
#     pin_memory: bool = True  # NOTE: Set to True by default instead of False.
#     drop_last: bool = False

#     __call__ = instantiate


def just(object_path: str):
    parts = object_path.split(".")
    mod = importlib.import_module(parts[0])
    for part in parts[1:]:
        mod = getattr(mod, part)
    return mod


def from_datasets(
    *args, num_classes: int | None = None, dims: tuple[int, ...] | None = None, **kwargs
):
    datamodule = LightningDataModule.from_datasets(*args, **kwargs)
    if num_classes is not None:
        datamodule.num_classes = num_classes  # type: ignore
    if dims is not None:
        datamodule.dims = dims  # type: ignore
    return datamodule


# datamodule_store(VisionDataModuleConfig, name="vision")

# inaturalist_config = hydra_zen.builds(
#     INaturalistDataModule,
#     builds_bases=(VisionDataModuleConfig,),
#     populate_full_signature=True,
#     dataclass_name=f"{INaturalistDataModule.__name__}Config",
# )
# datamodule_store(inaturalist_config, name="inaturalist")

from logging import getLogger as get_logger

from hydra_zen import store

logger = get_logger(__name__)

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


# datamodule_store(VisionDataModuleConfig, name="vision")

# inaturalist_config = hydra_zen.builds(
#     INaturalistDataModule,
#     builds_bases=(VisionDataModuleConfig,),
#     populate_full_signature=True,
#     dataclass_name=f"{INaturalistDataModule.__name__}Config",
# )
# datamodule_store(inaturalist_config, name="inaturalist")

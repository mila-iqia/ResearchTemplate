from . import algorithms, configs, datamodules, experiment, main, networks, utils
from .configs import Config, add_configs_to_hydra_store
from .experiment import Experiment
from .utils.hydra_utils import patched_safe_name  # noqa

# from .networks import FcNet
from .utils.types import DataModule

add_configs_to_hydra_store()


__all__ = [
    "algorithms",
    "experiment",
    "main",
    "Experiment",
    "configs",
    "datamodules",
    "networks",
    "DataModule",
    "utils",
    # "ExampleAlgorithm",
    # "ManualGradientsExample",
    # "NoOp",
    "Config",
    # "ImageClassificationDataModule",
    "DataModule",
    # "VisionDataModule",
    "Experiment",
    # "FcNet",
]

from hydra.core.config_store import ConfigStore
from project.algorithms.ppo.ppo import PPO
from .algorithm import Algorithm
from .backprop import Backprop
from .image_classification import ImageClassificationAlgorithm
from .manual_optimization_example import ManualGradientsExample
from .rl_example.reinforce import ExampleRLAlgorithm

# Store the different configuration options for each algorithm.

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.

_cs = ConfigStore.instance()
# _cs.store(group="algorithm", name="algorithm", node=Algorithm.HParams())


_cs.store(group="algorithm", name="backprop", node=Backprop.HParams())

# @hydrated_dataclass(target=Backprop, hydra_convert="object")
# class BackpropConfig:
#     # datamodule: Any = interpolated_field("${datamodule}")
#     # network: Any = interpolated_field("${network}")
#     hp: Backprop.HParams = field(default_factory=Backprop.HParams)
# _cs.store(group="algorithm", name="backprop", node=BackpropConfig)

# @hydrated_dataclass(target=PPO, hydra_convert="object")
# class PpoConfig:
#     # datamodule: Any = interpolated_field("${datamodule}")
#     # network: Any = interpolated_field("${network}")
#     hp: PPO.HParams = field(default_factory=PPO.HParams)
# _cs.store(group="algorithm", name="ppo", node=PpoConfig)


_cs.store(group="algorithm", name="manual_optimization", node=ManualGradientsExample.HParams())
_cs.store(group="algorithm", name="rl_example", node=ExampleRLAlgorithm.HParams())
_cs.store(group="algorithm", name="ppo", node=PPO.HParams())

__all__ = [
    "Algorithm",
    "Backprop",
    "ExampleRLAlgorithm",
    "ImageClassificationAlgorithm",
    "ManualGradientsExample",
]

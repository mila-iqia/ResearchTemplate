from hydra_zen import builds, store

from project.algorithms.no_op import NoOp
from project.algorithms.ppo.ppo import PPO

from .backprop import Backprop
from .bases.algorithm import Algorithm
from .bases.image_classification import ImageClassificationAlgorithm
from .manual_optimization_example import ManualGradientsExample
from .rl_example.reinforce import ExampleRLAlgorithm

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.

algorithm_store = store(group="algorithm")
algorithm_store(Backprop.HParams(), name="backprop")
algorithm_store(ManualGradientsExample.HParams(), name="manual_optimization")
algorithm_store(ExampleRLAlgorithm.HParams(), name="rl_example")
algorithm_store(PPO.HParams(), name="ppo")
algorithm_store(builds(NoOp, populate_full_signature=False), name="no_op")

# from hydra.core.config_store import ConfigStore
# cs = ConfigStore.instance()
# cs.store(group="algorithm", name="backprop", node=Backprop.HParams())
# cs.store(group="algorithm", name="manual_optimization", node=ManualGradientsExample.HParams())
# cs.store(group="algorithm", name="rl_example", node=ExampleRLAlgorithm.HParams())
# cs.store(group="algorithm", name="ppo", node=PPO.HParams())
# Store the different configuration options.

# from hydra_zen import hydrated_dataclass


# @hydrated_dataclass(target=Backprop, hydra_convert="object")
# class BackpropConfig:
#     __partial__: bool = True
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


__all__ = [
    "Algorithm",
    "Backprop",
    "ExampleRLAlgorithm",
    "ImageClassificationAlgorithm",
    "ManualGradientsExample",
]

from hydra_zen import builds, store

from project.algorithms.jax_algo import JaxAlgorithm
from project.algorithms.no_op import NoOp

from .bases.algorithm import Algorithm
from .bases.image_classification import ImageClassificationAlgorithm
from .example_algo import ExampleAlgorithm
from .manual_optimization_example import ManualGradientsExample

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.

algorithm_store = store(group="algorithm")
algorithm_store(ExampleAlgorithm.HParams(), name="example_algo")
algorithm_store(ManualGradientsExample.HParams(), name="manual_optimization")
algorithm_store(builds(NoOp, populate_full_signature=False), name="no_op")
algorithm_store(JaxAlgorithm.HParams(), name="jax_algo")

algorithm_store.add_to_hydra_store()

__all__ = [
    "Algorithm",
    "ExampleAlgorithm",
    "ImageClassificationAlgorithm",
    "ManualGradientsExample",
]

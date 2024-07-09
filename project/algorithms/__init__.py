from hydra_zen import builds, store

from project.algorithms.jax_example import JaxExample
from project.algorithms.no_op import NoOp

from .example import ExampleAlgorithm

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.
# todo: It might be nicer if we did this this `configs/algorithms` instead of here, no?
algorithm_store = store(group="algorithm")
algorithm_store(ExampleAlgorithm.HParams(), name="example_algo")
algorithm_store(builds(NoOp, populate_full_signature=False), name="no_op")
algorithm_store(JaxExample.HParams(), name="jax_example")

algorithm_store.add_to_hydra_store()

__all__ = [
    "ExampleAlgorithm",
    "ManualGradientsExample",
    "JaxExample",
]

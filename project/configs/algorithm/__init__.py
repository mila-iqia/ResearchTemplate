from hydra_zen import make_custom_builds_fn, store
from hydra_zen.third_party.pydantic import pydantic_parser

builds_fn = make_custom_builds_fn(
    zen_partial=True, populate_full_signature=True, zen_wrappers=pydantic_parser
)


# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and can use these configs in their default list.
algorithm_store = store(group="algorithm")


def populate_algorithm_store():
    # Note: import here to avoid circular imports.
    from project.algorithms import ExampleAlgorithm, JaxExample, NoOp

    algorithm_store(builds_fn(ExampleAlgorithm), name="example_algo")
    algorithm_store(builds_fn(NoOp), name="no_op")
    algorithm_store(builds_fn(JaxExample), name="jax_example")

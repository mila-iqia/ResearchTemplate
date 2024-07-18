from logging import getLogger as get_logger

import hydra_zen
from hydra_zen import make_custom_builds_fn, store
from lightning import LightningModule

from project.utils.hydra_utils import make_config_and_store

from .lr_scheduler import add_configs_for_all_torch_schedulers, lr_scheduler_store
from .optimizer import add_configs_for_all_torch_optimizers, optimizers_store

logger = get_logger(__name__)

build_algorithm_config_fn = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=True,
    zen_exclude=["datamodule", "network"],
)


# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `project/configs/algorithm`, it will also be available as an option
# from the command-line, and can use these configs in their default list.
algorithm_store = store(group="algorithm")


def register_algorithm_configs(with_dynamic_configs: bool = True):
    if with_dynamic_configs:
        add_configs_for_all_torch_optimizers()
        add_configs_for_all_torch_schedulers()

    optimizers_store.add_to_hydra_store()
    lr_scheduler_store.add_to_hydra_store()

    import inspect

    # Note: import algorithms here to avoid circular import errors.
    import project.algorithms

    for algo_name, algo_class in [
        (k, v)
        for (k, v) in vars(project.algorithms).items()
        if inspect.isclass(v) and issubclass(v, LightningModule)
    ]:
        config = make_config_and_store(
            algo_class, store=algorithm_store, zen_exclude=["datamodule", "network"]
        )
        logger.info(f"Registered config for {algo_name}: {config}")
        assert hydra_zen.get_target(config) == algo_class
        # config_class_name = f"{algo_name}Config"
        # config_class = build_algorithm_config_fn(
        #     algo_class, zen_dataclass={"cls_name": config_class_name}
        # )
        # algorithm_store(config_class, name=algo_name)

    # from project.algorithms import ExampleAlgorithm, JaxExample, NoOp
    # algorithm_store(build_algorithm_config_fn(ExampleAlgorithm), name="example")
    # algorithm_store(build_algorithm_config_fn(NoOp), name="no_op")
    # algorithm_store(build_algorithm_config_fn(JaxExample), name="jax_example")

    algorithm_store.add_to_hydra_store()

from logging import getLogger as get_logger

from hydra_zen import make_custom_builds_fn, store

from .lr_scheduler import add_configs_for_all_torch_schedulers, lr_scheduler_store
from .optimizer import add_configs_for_all_torch_optimizers, optimizers_store

logger = get_logger(__name__)

# note; Can also create configs programmatically with this.
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

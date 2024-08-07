"""Configs for optimizers.

By default, this adds configuration classes for all the optimizers in `torch.optim`.
This is done using [hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""

import inspect
from logging import getLogger as get_logger

import hydra_zen
import torch
import torch.optim
from hydra_zen.typing import PartialBuilds

from project.utils.hydra_utils import make_config_and_store

_OPTIMIZER_GROUP = "algorithm/optimizer"

_logger = get_logger(__name__)

optimizers_store = hydra_zen.ZenStore(name="optimizers")
optimizers_group_store = optimizers_store(group=_OPTIMIZER_GROUP)


# Create some configs manually so they can get nice type hints when imported.
AdamConfig = make_config_and_store(torch.optim.Adam, store=optimizers_group_store)


SGDConfig = make_config_and_store(torch.optim.SGD, store=optimizers_group_store)


def add_configs_for_all_torch_optimizers():
    """Generates configuration dataclasses for all `torch.optim.Optimizer` classes that are not
    already configured.

    Registers the configs using the `make_config_and_store` function.
    """
    configured_schedulers = [
        hydra_zen.get_target(config) for config in get_all_optimizer_configs()
    ]
    missing_torch_schedulers = {
        _name: _optimizer_type
        for _name, _optimizer_type in vars(torch.optim).items()
        if inspect.isclass(_optimizer_type)
        and issubclass(_optimizer_type, torch.optim.Optimizer)
        and _optimizer_type is not torch.optim.Optimizer
        and _optimizer_type not in configured_schedulers
    }
    for scheduler_name, scheduler_type in missing_torch_schedulers.items():
        _logger.debug(f"Making a config for {scheduler_type=}")
        _config = make_config_and_store(scheduler_type, store=optimizers_group_store)


def get_all_config_names() -> list[str]:
    return sorted(
        [config_name for (_group, config_name) in optimizers_store[_OPTIMIZER_GROUP].keys()]
    )


def get_all_optimizer_configs() -> list[type[PartialBuilds[torch.optim.Optimizer]]]:
    return list(optimizers_store[_OPTIMIZER_GROUP].values())

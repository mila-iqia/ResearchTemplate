"""Configs for learning rate schedulers from `torch.optim.lr_scheduler`.

These configurations are created dynamically using [hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""

import inspect
from logging import getLogger as get_logger

import hydra_zen
import torch
import torch.optim.lr_scheduler
from hydra_zen.typing import PartialBuilds

from project.utils.hydra_utils import make_config_and_store

_logger = get_logger(__name__)

_LR_SCHEDULER_GROUP = "algorithm/lr_scheduler"
lr_scheduler_store = hydra_zen.ZenStore(name="schedulers")

lr_scheduler_group_store = lr_scheduler_store(group=_LR_SCHEDULER_GROUP)

# Some LR Schedulers have constructors with arguments without a default value (in addition to optimizer).
# In this case, we specify the missing arguments here so we get a nice error message if it isn't passed.

CosineAnnealingLRConfig = make_config_and_store(
    torch.optim.lr_scheduler.CosineAnnealingLR,
    T_max="???",
    store=lr_scheduler_group_store,
)

StepLRConfig = make_config_and_store(
    torch.optim.lr_scheduler.StepLR,
    step_size="???",
    store=lr_scheduler_group_store,
)


def add_configs_for_all_torch_schedulers():
    """Generates configuration dataclasses for all torch.optim.lr_scheduler classes that are not
    already configured.

    Registers the configs using the `make_config_and_store` function.
    """
    configured_schedulers = [
        hydra_zen.get_target(config) for config in get_all_scheduler_configs()
    ]
    missing_torch_schedulers = {
        _name: _scheduler_type
        for _name, _scheduler_type in vars(torch.optim.lr_scheduler).items()
        if inspect.isclass(_scheduler_type)
        and issubclass(_scheduler_type, torch.optim.lr_scheduler.LRScheduler)
        and _scheduler_type
        not in (torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler._LRScheduler)
        and _scheduler_type not in configured_schedulers
    }
    for scheduler_name, scheduler_type in missing_torch_schedulers.items():
        _logger.warning(f"Making a config for {scheduler_type=}")
        _config = make_config_and_store(scheduler_type, store=lr_scheduler_group_store)


def get_all_config_names() -> list[str]:
    return sorted(
        [config_name for (_group, config_name) in lr_scheduler_store[_LR_SCHEDULER_GROUP].keys()]
    )


def get_all_scheduler_configs() -> list[type[PartialBuilds[torch.optim.lr_scheduler.LRScheduler]]]:
    return list(lr_scheduler_store[_LR_SCHEDULER_GROUP].values())


# def __getattr__(config_name: str) -> type[PartialBuilds[torch.optim.lr_scheduler.LRScheduler]]:
#     """Get the dynamically generated LR scheduler config with the given name."""
#     if config_name in globals():
#         return globals()[config_name]
#     if not config_name.endswith("Config"):
#         raise AttributeError
#     scheduler_name = config_name.removesuffix("Config")
#     # the keys for the config store are tuples of the form (group, config_name)
#     store_key = (_LR_SCHEDULER_GROUP, scheduler_name)
#     if store_key in lr_scheduler_store[_LR_SCHEDULER_GROUP]:
#         _logger.debug(f"Dynamically retrieving the config for {scheduler_name!r}")
#         return lr_scheduler_store[store_key]
#     available_configs = sorted(
#         config_name for (_group, config_name) in lr_scheduler_store[_LR_SCHEDULER_GROUP].keys()
#     )
#     _logger.error(
#         f"Unable to find the config for {scheduler_name=}. Available configs: {available_configs}."
#     )

#     raise AttributeError

"""Configs for learning rate schedulers from `torch.optim.lr_scheduler`.

These configurations are created dynamically using [hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""

import inspect
from logging import getLogger as get_logger

import hydra_zen
import torch
import torch.optim.lr_scheduler

_logger = get_logger(__name__)


# Some LR Schedulers have constructors with arguments without a default value (in addition to optimizer).
# In this case, we specify the missing arguments here so we get a nice error message if it isn't passed.

# StepLRConfig = hydra_zen.builds(
#     torch.optim.lr_scheduler.StepLR,
#     populate_full_signature=True,
#     step_size="???",
#     zen_partial=True,
#     zen_dataclass={"cls_name": "StepLRConfig", "frozen": True},
# ),
# network_store(ResNet18Config, name="resnet18")


def add_configs_for_all_torch_schedulers():
    """Generates configuration classes for all the LR schedulers in `torch.optim.lr_scheduler`."""

    _LR_SCHEDULER_GROUP = "algorithm/lr_scheduler"

    lr_scheduler_store = hydra_zen.ZenStore(name="schedulers")

    configured_schedulers = [
        hydra_zen.get_target(config) for config in lr_scheduler_store[_LR_SCHEDULER_GROUP].values()
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
        _logger.debug(f"Making a config for {scheduler_type=}")
        # Create a config programmatically.
        config_class = hydra_zen.builds(
            scheduler_type,
            zen_partial=True,
            populate_full_signature=True,
            zen_dataclass={"cls_name": f"{scheduler_name}Config", "frozen": True},
        )
        # Add the config to the config store.
        lr_scheduler_store(config_class, name=scheduler_name, group=_LR_SCHEDULER_GROUP)

    # Add the configs to the Hydra config store.
    lr_scheduler_store.add_to_hydra_store(overwrite_ok=False)

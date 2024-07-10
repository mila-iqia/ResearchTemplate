"""Configs for learning rate schedulers from `torch.optim.lr_scheduler`.

These configurations are created dynamically using [hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""

import dataclasses
import inspect
from logging import getLogger as get_logger

import torch
import torch.optim.lr_scheduler
from hydra_zen import make_custom_builds_fn, store
from hydra_zen.typing import PartialBuilds

_logger = get_logger(__name__)

_LR_SCHEDULER_GROUP = "algorithm/lr_scheduler"
lr_scheduler_store = store(group=_LR_SCHEDULER_GROUP)


_build_scheduler_config = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


# Some LR Schedulers have constructors with arguments without a default value (in addition to optimizer).
# In this case, we specify the missing arguments here so we get a nice error message if it isn't passed.
CosineAnnealingLRConfig = _build_scheduler_config(
    torch.optim.lr_scheduler.CosineAnnealingLR,
    T_max="???",
    zen_dataclass={"cls_name": "CosineAnnealingLRConfig"},
)
lr_scheduler_store(CosineAnnealingLRConfig, name="CosineAnnealingLR")


StepLRConfig = _build_scheduler_config(
    torch.optim.lr_scheduler.StepLR,
    step_size="???",
    zen_dataclass={"cls_name": "StepLRConfig"},
)
lr_scheduler_store(StepLRConfig, name="StepLR")


_configs_defined_so_far = [k for k, v in locals().items() if dataclasses.is_dataclass(v)]
for scheduler_name, scheduler_type in [
    (_name, _obj)
    for _name, _obj in vars(torch.optim.lr_scheduler).items()
    if inspect.isclass(_obj)
    and issubclass(_obj, torch.optim.lr_scheduler.LRScheduler)
    and _obj not in (torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler._LRScheduler)
]:
    _config_name = f"{scheduler_name}Config"
    if _config_name in _configs_defined_so_far:
        # We already have a hand-made config for this scheduler. Skip it.
        continue

    _lr_scheduler_config = _build_scheduler_config(
        scheduler_type, zen_dataclass={"cls_name": _config_name}
    )
    lr_scheduler_store(_lr_scheduler_config, name=scheduler_name)
    _logger.debug(f"Registering a config for {scheduler_type=}")


def __getattr__(config_name: str) -> type[PartialBuilds[torch.optim.lr_scheduler.LRScheduler]]:
    """Get the dynamically generated LR scheduler config with the given name."""
    if not config_name.endswith("Config"):
        raise AttributeError
    scheduler_name = config_name.removesuffix("Config")
    # the keys for the config store are tuples of the form (group, config_name)
    store_key = (_LR_SCHEDULER_GROUP, scheduler_name)
    if store_key in lr_scheduler_store[_LR_SCHEDULER_GROUP]:
        _logger.debug(f"Dynamically retrieving the config for {scheduler_name!r}")
        return lr_scheduler_store[store_key]
    available_configs = sorted(
        config_name for (_group, config_name) in lr_scheduler_store[_LR_SCHEDULER_GROUP].keys()
    )
    _logger.error(
        f"Unable to find the config for {scheduler_name=}. Available configs: {available_configs}."
    )

    raise AttributeError


def get_all_config_names() -> list[str]:
    return sorted(
        [config_name for (_group, config_name) in lr_scheduler_store[_LR_SCHEDULER_GROUP].keys()]
    )


def get_all_configs() -> list[type[PartialBuilds[torch.optim.lr_scheduler.LRScheduler]]]:
    return list(lr_scheduler_store[_LR_SCHEDULER_GROUP].values())

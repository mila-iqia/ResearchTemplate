import dataclasses
import inspect
from logging import getLogger as get_logger

import torch
import torch.optim.lr_scheduler
from hydra_zen import make_custom_builds_fn, store

logger = get_logger(__name__)

builds_fn = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# LR Schedulers whose constructors have arguments with missing defaults have to be created manually,
# because we otherwise get some errors if we try to use them (e.g. T_max doesn't have a default.)

CosineAnnealingLRConfig = builds_fn(torch.optim.lr_scheduler.CosineAnnealingLR, T_max="???")
StepLRConfig = builds_fn(torch.optim.lr_scheduler.StepLR, step_size="???")
lr_scheduler_store = store(group="algorithm/lr_scheduler")
lr_scheduler_store(StepLRConfig, name="StepLR")
lr_scheduler_store(CosineAnnealingLRConfig, name="CosineAnnealingLR")


# IDEA: Could be interesting to generate configs for any member of the torch.optimizer.lr_scheduler
# package dynamically (and store it)?
# def __getattr__(self, name: str):
#     """"""

_configs_defined_so_far = [k for k, v in locals().items() if dataclasses.is_dataclass(v)]
for scheduler_name, scheduler_type in [
    (_name, _obj)
    for _name, _obj in vars(torch.optim.lr_scheduler).items()
    if inspect.isclass(_obj)
    and issubclass(_obj, torch.optim.lr_scheduler.LRScheduler)
    and _obj is not torch.optim.lr_scheduler.LRScheduler
]:
    _config_name = f"{scheduler_name}Config"
    if _config_name in _configs_defined_so_far:
        # We already have a hand-made config for this scheduler. Skip it.
        continue

    _lr_scheduler_config = builds_fn(scheduler_type, zen_dataclass={"cls_name": _config_name})
    lr_scheduler_store(_lr_scheduler_config, name=scheduler_name)
    logger.debug(f"Registering config for the {scheduler_type} LR scheduler.")


def __getattr__(config_name: str):
    if not config_name.endswith("Config"):
        raise AttributeError
    scheduler_name = config_name.removesuffix("Config")
    # the keys for the config store are tuples of the form (group, config_name)
    group = "algorithm/lr_scheduler"
    store_key = (group, scheduler_name)
    if store_key in lr_scheduler_store[group]:
        logger.debug(f"Dynamically retrieving the config for the {scheduler_name} LR scheduler.")
        return lr_scheduler_store[store_key]
    available_configs = sorted(
        config_name for (_group, config_name) in lr_scheduler_store[group].keys()
    )
    logger.error(
        f"Unable to find the config for {scheduler_name=}. Available configs: {available_configs}."
    )

    raise AttributeError

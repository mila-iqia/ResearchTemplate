import inspect
from logging import getLogger as get_logger

import torch
import torch.optim
from hydra_zen import make_custom_builds_fn, store
from hydra_zen.typing import PartialBuilds

_OPTIMIZER_GROUP = "algorithm/optimizer"

_logger = get_logger(__name__)
_build_optimizer_config_fn = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

optimizer_store = store(group=_OPTIMIZER_GROUP)

# Note: instead of creating configs for each optimizer manually like this:
# AdamConfig = builds_fn(torch.optim.Adam)
# SGDConfig = builds_fn(torch.optim.SGD)
# optimizer_store(AdamConfig, name="adam")
# optimizer_store(SGDConfig, name="sgd")

# We can dynamically create configs for all optimizers in torch.optim like so:
for optimizer_name, optimizer_type in [
    (k, v)
    for k, v in vars(torch.optim).items()
    if inspect.isclass(v)
    and issubclass(v, torch.optim.Optimizer)
    and v is not torch.optim.Optimizer
]:
    _algo_config = _build_optimizer_config_fn(
        optimizer_type, zen_dataclass={"cls_name": f"{optimizer_name}Config"}
    )
    optimizer_store(_algo_config, name=optimizer_name)
    _logger.debug(f"Registering config for {optimizer_type=}.")


def __getattr__(config_name: str) -> type[PartialBuilds[torch.optim.Optimizer]]:
    """Get the optimizer config with the given name."""

    if not config_name.endswith("Config"):
        raise AttributeError
    optimizer_name = config_name.removesuffix("Config")
    # the keys for the config store are tuples of the form (group, config_name)
    store_key = (_OPTIMIZER_GROUP, optimizer_name)
    if store_key in optimizer_store[_OPTIMIZER_GROUP]:
        _logger.debug(f"Dynamically retrieving the config for {optimizer_name=}.")
        return optimizer_store[store_key]
    available_optimizers = sorted(
        optimizer_name for (_, optimizer_name) in optimizer_store[_OPTIMIZER_GROUP].keys()
    )
    _logger.error(
        f"Unable to find the config for optimizer {optimizer_name}. Available optimizers: {available_optimizers}."
    )

    raise AttributeError


def get_all_config_names() -> list[str]:
    return sorted(
        [config_name for (_group, config_name) in optimizer_store[_OPTIMIZER_GROUP].keys()]
    )


def get_all_configs() -> list[type[PartialBuilds[torch.optim.Optimizer]]]:
    return list(optimizer_store[_OPTIMIZER_GROUP].values())

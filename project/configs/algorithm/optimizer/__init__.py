import inspect
from logging import getLogger as get_logger

import torch
import torch.optim
from hydra_zen import make_custom_builds_fn, store

logger = get_logger(__name__)
builds_fn = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

optimizer_store = store(group="algorithm/optimizer")
# AdamConfig = builds_fn(torch.optim.Adam)
# SGDConfig = builds_fn(torch.optim.SGD)
# optimizer_store(AdamConfig, name="adam")
# optimizer_store(SGDConfig, name="sgd")

for optimizer_name, optimizer_type in [
    (k, v)
    for k, v in vars(torch.optim).items()
    if inspect.isclass(v)
    and issubclass(v, torch.optim.Optimizer)
    and v is not torch.optim.Optimizer
]:
    _algo_config = builds_fn(optimizer_type, zen_dataclass={"cls_name": f"{optimizer_name}Config"})
    optimizer_store(_algo_config, name=optimizer_name)
    logger.debug(f"Registering config for the {optimizer_type} optimizer.")


def __getattr__(config_name: str):
    if not config_name.endswith("Config"):
        raise AttributeError
    optimizer_name = config_name.removesuffix("Config")
    # the keys for the config store are tuples of the form (group, config_name)
    store_key = ("algorithm/optimizer", optimizer_name)
    if store_key in optimizer_store["algorithm/optimizer"]:
        logger.debug(f"Dynamically retrieving the config for the {optimizer_name} optimizer.")
        return optimizer_store[store_key]
    available_optimizers = sorted(
        optimizer_name for (_, optimizer_name) in optimizer_store["algorithm/optimizer"].keys()
    )
    logger.error(
        f"Unable to find the config for optimizer {optimizer_name}. Available optimizers: {available_optimizers}."
    )

    raise AttributeError

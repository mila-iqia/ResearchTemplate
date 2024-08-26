"""Configurations for optimizers.

You can add configurations either with a config file or in code using
[hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""

import hydra_zen
import torch
import torch.optim

# NOTE: Can also create configs programmatically with hydra-zen.
# This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# Create some configs manually so they can get nice type hints when imported.
AdamConfig = hydra_zen.builds(
    # note: getting this 'Adam is not exported from `torch.optim`' typing error, but importing it
    # from torch.optim.adam doesn't work (because they del the `adam` module in torch.optim!)
    torch.optim.Adam,  # type: ignore
    zen_partial=True,
    populate_full_signature=True,
    zen_dataclass={"cls_name": "AdamConfig", "frozen": True},
)

SGDConfig = hydra_zen.builds(
    torch.optim.SGD,  # type: ignore
    zen_partial=True,
    populate_full_signature=True,
    zen_dataclass={"cls_name": "SGDConfig", "frozen": True},
)

# If you add a configuration file under `project/configs/algorithm`, it will also be available as an option
# from the command-line, and can use these configs in their default list.
optimizers_store = hydra_zen.store(group="optimizer")
# NOTE: You can also add your configs to the config store programmatically like this instead of
# adding a config file:

# store the config in the config group.
# optimizers_store(AdamConfig, name="Adam")
# optimizers_store(SGDConfig, name="SGD")

"""Configurations for optimizers.

You can add configurations either with a config file or by registering structured configs in code.

Here is an example of how you could register a new configuration in code using
[hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#):


```python
import hydra_zen
from torch.optim import Adam  # type: ignore

optimizers_store = hydra_zen.store(group="algorithm/optimizer")

AdamConfig = optimizers_store(
    hydra_zen.builds(
        Adam,
        zen_partial=True,
        populate_full_signature=True,
        zen_exclude=["params"],
        zen_dataclass={"cls_name": "AdamConfig", "frozen": False},
    ),
    name="base_adam",
)
```

From the command-line, you can select both configs that are yaml files as well as structured config
(dataclasses).

This works the same way as creating config files for each optimizer under `configs/algorithm/optimizer`.
Config files can also use structured configs in their defaults list.
"""

import hydra_zen

optimizers_store = hydra_zen.store(group="algorithm/optimizer")

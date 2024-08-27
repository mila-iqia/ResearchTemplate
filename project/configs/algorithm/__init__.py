"""Configs for algorithms."""

import hydra_zen

# note; Can also create configs programmatically with hydra-zen.
# This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `project/configs/algorithm`, it will also be available as an option
# from the command-line, and can use these configs in their default list.
algorithm_store = hydra_zen.store(group="algorithm")

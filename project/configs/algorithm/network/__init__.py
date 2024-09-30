from hydra_zen import store

network_store = store(group="network")
# note; Can also create configs programmatically with hydra-zen.
# This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `project/configs/network`, it will also be available as an option
# from the command-line, and can use these configs in their default list.
# network_store = hydra_zen.store(group="network")

# ResNet18Config = hydra_zen.builds(
#     torchvision.models.resnet18,
#     populate_full_signature=True,
#     num_classes="${instance_attr:datamodule.num_classes:1000}",
# ),
# network_store(ResNet18Config, name="resnet18")

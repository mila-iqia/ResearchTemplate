# Design problem: How we create the network depends on the kind of datamodule (and later on maybe
# even Algorithm..) that we use.
# Option 1: Create a common interface (e.g. have DataModule have input_shape/space and output_shape
# or similar)
# Option 2: Create handlers for each kind of datamodule (e.g. VisionDataModule, RLDataModule, ...)
# using something like Singledispatch:
# - handler for creating the network from a VisionDataModule
# - handler for creating the network from an RLDataModule
# - ...
# Currently, we're using something like option 1, where we use `interpolated_field` to retrieve
# some attributes from the datamodule when creating the network configs.
# _cs = ConfigStore.instance()
# _cs.store(group="network", name="fcnet", node=FcNetConfig)
# _cs.store(group="network", name="resnet18", node=ResNet18Config)
# Add your network configs here.

from .fcnet import FcNet

__all__ = ["FcNet"]

from dataclasses import field

from hydra.core.config_store import ConfigStore
from hydra_zen import hydrated_dataclass
from torchvision.models import resnet18

from project.utils.hydra_utils import interpolated_field

from .fcnet import FcNet


@hydrated_dataclass(target=resnet18)
class ResNet18Config:
    pretrained: bool = False
    num_classes: int = interpolated_field(
        "${instance_attr:datamodule.num_classes,datamodule.action_dims}", default=1000
    )


@hydrated_dataclass(target=FcNet, hydra_recursive=True, hydra_convert="object")
class FcNetConfig:
    output_dims: int = interpolated_field(
        "${instance_attr:datamodule.num_classes,datamodule.action_dims}", default=1
    )
    hparams: FcNet.HParams = field(default_factory=FcNet.HParams)


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

_cs = ConfigStore.instance()
_cs.store(group="network", name="fcnet", node=FcNetConfig)
_cs.store(group="network", name="resnet18", node=ResNet18Config)
...
# Add your network configs here.

__all__ = ["FcNet", "FcNetConfig", "ResNet18Config"]

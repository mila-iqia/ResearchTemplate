from hydra.core.config_store import ConfigStore
from project.networks.fcnet import FcNet
from torchvision.models import resnet18
from dataclasses import field
from hydra_zen import hydrated_dataclass
from project.utils.hydra_utils import interpolate_or_default


@hydrated_dataclass(target=resnet18)
class ResNet18Config:
    pretrained: bool = False
    num_classes: int = interpolate_or_default("${datamodule:num_classes}", 1000)


@hydrated_dataclass(target=FcNet, hydra_recursive=True, hydra_convert="object")
class FcNetConfig:
    input_shape: tuple[int, ...] = interpolate_or_default("${datamodule:dims}", (3, 32, 32))
    output_shape: int = interpolate_or_default("${datamodule:num_classes}", (10))
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

_cs = ConfigStore.instance()
_cs.store(group="network", name="fcnet", node=FcNetConfig)
# _cs.store(group="network", name="fcnet", node=FcNet.HParams)
_cs.store(group="network", name="resnet18", node=ResNet18Config)

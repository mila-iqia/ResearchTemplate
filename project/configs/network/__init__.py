import hydra_zen
import torchvision.models
from hydra_zen import store

from project.networks.fcnet import FcNet
from project.utils.hydra_utils import interpolate_config_attribute

network_store = store(group="network")
network_store(
    hydra_zen.builds(
        torchvision.models.resnet18,
        populate_full_signature=True,
        num_classes=interpolate_config_attribute("datamodule.num_classes"),
    ),
    name="resnet18",
)
network_store(
    hydra_zen.builds(
        FcNet,
        hydra_convert="object",
        hydra_recursive=True,
        populate_full_signature=True,
        output_dims=interpolate_config_attribute("datamodule.num_classes"),
    ),
    name="fcnet",
)

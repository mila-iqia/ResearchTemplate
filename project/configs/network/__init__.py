from hydra_zen import store

from project.networks import FcNetConfig, ResNet18Config

network_store = store(group="network")
network_store(FcNetConfig, name="fcnet")
network_store(ResNet18Config, name="resnet18")

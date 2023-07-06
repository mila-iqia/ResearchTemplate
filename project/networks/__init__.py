from hydra.core.config_store import ConfigStore
from .fcnet import FcNet

_cs = ConfigStore.instance()
# _cs.store(group="network", name="network", node=Network.HParams())
# _cs.store(group="network", name="simple_vgg", node=SimpleVGG.HParams())
# _cs.store(group="network", name="lenet", node=LeNet.HParams())
# _cs.store(group="network", name="resnet18", node=ResNet18.HParams())
# _cs.store(group="network", name="resnet34", node=ResNet34.HParams())
_cs.store(group="network", name="fcnet", node=FcNet.HParams())

__all__ = []

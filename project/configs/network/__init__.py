import inspect
from typing import TypeVar, cast
from dataclasses import field
import functools
from hydra_zen import hydrated_dataclass, instantiate
from torchvision.models import resnet18
from hydra.core.config_store import ConfigStore
from omegaconf import SI, II

T = TypeVar("T")


def _default_factory_(interpolation: str, default_val: T) -> T:
    current = inspect.currentframe()
    assert current
    prev = current.f_back
    assert prev
    if inspect.getframeinfo(prev).function == "get_dataclass_data":
        return cast(T, interpolation)  # type: ignore
    return default_val
    caller_functions = []
    while prev:
        caller_functions.append(inspect.getframeinfo(prev))
        prev = prev.f_back
        assert False, caller_functions[0]
    assert False, caller_functions


def interpolate_or_default(interpolation: str, default: T) -> T:
    # TODO: If we're in a Hydra instantiate context, return the variable interpolation default (the string)
    # otherwise, we're probably in the regular dataclass context, so return the default value.
    assert "${" in interpolation and "}" in interpolation
    return field(default_factory=functools.partial(_default_factory_, interpolation, default))


@hydrated_dataclass(target=resnet18)
class ResNet18Config:
    pretrained: bool = False
    num_classes: int = interpolate_or_default("${datamodule:num_classes}", 1000)


_cs = ConfigStore.instance()
_cs.store("resnet18", group="network", node=ResNet18Config)

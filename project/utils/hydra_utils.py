from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from hydra_zen.typing._implementations import Partial as _Partial
from typing_extensions import ParamSpec

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")


def config_name(target_type: type) -> str:
    return target_type.__name__ + "Config"


def get_config(group: str, name: str):
    cs = ConfigStore.instance()
    return cs.load(f"{group}/{name}.yaml").node


# @dataclass(init=False)
class Partial(functools.partial[T], _Partial[T]):
    def __getattr__(self, name: str):
        if name in self.keywords:
            return self.keywords[name]
        raise AttributeError(name)


def add_attributes(fn: functools.partial[T]) -> Partial[T]:
    """Adds a __getattr__ to the partial that returns the value in `v.keywords`."""
    if isinstance(fn, Partial):
        return fn
    assert isinstance(fn, functools.partial)
    return Partial(fn.func, *fn.args, **fn.keywords)


# BUG: Getting a weird annoying error with omegaconf if I try to make this generic!
@dataclass
class CallableConfig:
    """Little mixin that makes it possible to call the config, like adam_config() -> Adam, instead
    of having to use `instantiate`."""

    def __call__(self, *args, **kwargs):
        object = instantiate(self, *args, **kwargs)
        if isinstance(object, functools.partial):
            if isinstance(object, Partial):
                return object
            # Return out own Partial class, so that we can access the keywords as attributes.
            return Partial(object.func, *object.args, **object.keywords)
        return object


@runtime_checkable
class ConfigDataclass(Protocol[T]):
    def __call__(self, *args, **kwargs) -> T:
        ...


def is_inner_class(object_type: type) -> bool:
    return "." in object_type.__qualname__


def get_full_name(object_type: type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_outer_class(inner_class: type) -> type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    # assert False, (container_class_module, class_name, child_name)
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


class_to_config_class: dict[type, type] = {}

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from collections import ChainMap
from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Literal, TypeVar

from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from hydra_zen.typing._implementations import Partial as _Partial
from typing_extensions import ParamSpec

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")


def get_instantiated_attr(*attributes: str):
    """Quite hacky: Allows interpolations to get the value of the objects, rather than configs."""
    if not attributes:
        raise RuntimeError("Need to pass one or more attributes to this resolver.")

    frame = inspect.currentframe()
    assert frame
    frame = frame.f_back
    frames = []
    frame_infos = []

    while frame:
        frame_info = inspect.getframeinfo(frame)
        frames.append(frame)
        frame_infos.append(frame_info)
        frame = frame.f_back

    # These local variables are defined in the _to_object function inside OmegaConf.
    init_field_items: list[dict[str, Any]] = []
    non_init_field_items: list[dict[str, Any]] = []
    for frame in frames:
        if "init_field_items" in frame.f_locals:
            init_field_items.append(frame.f_locals["init_field_items"].copy())
        if "non_init_field_items" in frame.f_locals:
            non_init_field_items.append(frame.f_locals["non_init_field_items"].copy())

    # Okay so we now have all the *instantiated* attributes! We can do the interpolation by
    # getting its value!

    all_init_field_items = ChainMap(*reversed(init_field_items))
    for attribute in attributes:
        key, *nested_attribute = attribute.split(".")
        if key not in all_init_field_items:
            continue
        obj = all_init_field_items[key]
        for attr_part in nested_attribute:
            if not hasattr(obj, attr_part):
                break
            obj = getattr(obj, attr_part)
        else:
            # Found the attribute!
            return obj

    raise RuntimeError(
        f"Could not find any attributes matching {attributes} on in the instantiated objects.\n"
        f"The following attributes were found:\n"
        + "\n".join([f"- {k}: {type(v)}" for k, v in all_init_field_items.items()])
    )


def interpolated_field(
    interpolation: str,
    default: T | Literal[MISSING] = MISSING,
    default_factory: Callable[[], T] | Literal[MISSING] = MISSING,
) -> T:
    """Returns the string for interpolation when in a Hydra instantiate context, otherwise default.

    This is sort-of hacky.
    """
    assert "${" in interpolation and "}" in interpolation
    assert default is not MISSING or default_factory is not MISSING
    return field(
        default_factory=functools.partial(
            _default_factory,
            interpolation=interpolation,
            default=default,
            default_factory=default_factory,
        )
    )


def _default_factory(
    interpolation: str,
    default: T | Literal[dataclasses.MISSING] = dataclasses.MISSING,
    default_factory: Callable[[], T] | Literal[dataclasses.MISSING] = dataclasses.MISSING,
) -> T:
    current = inspect.currentframe()
    assert current
    prev = current.f_back
    assert prev
    if inspect.getframeinfo(prev).function == "get_dataclass_data":
        return interpolation  # type: ignore
    if default_factory is not dataclasses.MISSING:
        return default_factory()
    return default  # type: ignore


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

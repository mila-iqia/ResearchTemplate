from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from collections import ChainMap
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Callable, Literal, Mapping, MutableMapping, TypeVar
from logging import getLogger as get_logger

from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
from hydra_zen.typing._implementations import Partial as _Partial
from typing_extensions import ParamSpec

from omegaconf import DictConfig


logger = get_logger(__name__)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
P = ParamSpec("P")
R = TypeVar("R")


def get_attr(obj: Any, *attributes: str):
    if not attributes:
        return obj
    for attribute in attributes:
        subobj = obj
        try:
            for attr in attribute.split("."):
                subobj = getattr(subobj, attr)
            return subobj
        except AttributeError:
            pass
    raise AttributeError(f"Could not find any attributes matching {attributes} on {obj}.")


def has_attr(obj: Any, potentially_nested_attribute: str):
    for attribute in potentially_nested_attribute.split("."):
        if not hasattr(obj, attribute):
            return False
        obj = getattr(obj, attribute)
    return True


def set_attr(obj: Any, potentially_nested_attribute: str, value: Any) -> None:
    attributes = potentially_nested_attribute.split(".")
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def get_instantiated_attr(
    *attributes: str, _instantiated_objects_cache: MutableMapping[str, Any] | None = None
):
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

    # SUPER HACKY: These local variables are defined in the _to_object function inside OmegaConf.
    init_field_items: list[dict[str, Any]] = []
    non_init_field_items: list[dict[str, Any]] = []
    for frame in frames:
        if "init_field_items" in frame.f_locals:
            init_field_items.append(frame.f_locals["init_field_items"].copy())
        if "non_init_field_items" in frame.f_locals:
            non_init_field_items.append(frame.f_locals["non_init_field_items"].copy())

    # Okay so we now have all the *instantiated* attributes! We can do the interpolation by
    # getting its value!
    # BUG: This isn't quite working as I'd like, the interpolation is trying to get the attribute
    # on the object *instance*, but this is getting called during the first OmegaConf.to_object
    # which is only responsible for creating the configs, not the objects!
    all_init_field_items = ChainMap(*reversed(init_field_items))

    if _instantiated_objects_cache is not None:
        _instantiated_objects_cache.update(all_init_field_items)

    objects_cache: Mapping[str, Any] = _instantiated_objects_cache or all_init_field_items

    for attribute in attributes:
        logger.debug(f"Looking for attribute {attribute} from {all_init_field_items}")
        key, *nested_attribute = attribute.split(".")
        if key not in objects_cache:
            continue

        obj = objects_cache[key]

        # Recursive getattr that tries to instantiate configs to objects when missing an attribute
        # values.
        new_instantiated_objects: dict[str, Any] = {}  # NOTE: unused so far.
        for level, attr_part in enumerate(nested_attribute):
            if hasattr(obj, attr_part):
                obj = getattr(obj, attr_part)
                continue

            if not (
                (isinstance(obj, (dict, DictConfig)) and "_target_" in obj)
                or (is_dataclass(obj) and any(f.name == "_target_" for f in fields(obj)))
            ):
                # attribute not found, and the `obj` isn't a config with a _target_ field.
                break

            # FIXME: SUPER HACKY: Instantiating the object just to get the value. Super bad.
            # FIXME: It's either this, or we pull a surprise and replace the config object in
            # the cache with the instantiated object.

            path_so_far = key
            if nested_attribute[:level]:
                path_so_far += "." + ".".join(nested_attribute[:level])
            logger.debug(
                f"Will pro-actively attempt to instantiate {path_so_far} just to retrieve "
                f"the {attr_part} attribute."
            )
            try:
                instantiated_obj = instantiate(obj)
            except Exception as err:
                logger.debug(
                    f"Unable to instantiate {obj} to get the missing {attr_part} "
                    f"attribute: {err}"
                )
                break
            new_instantiated_objects[path_so_far] = instantiated_obj

            logger.info(
                f"Instantiated the config at {path_so_far!r} while trying to find one of the "
                f"{attributes} attributes."
            )

            if _instantiated_objects_cache is None:
                logger.warning(
                    f"The config {obj} at path {path_so_far} was instantiated while trying to "
                    f"find the {attr_part} portion of the {attribute} attributes in {attributes}."
                    f"This compute is being wasted because {_instantiated_objects_cache=}. "
                    f"If this is an expensive config to instantiate (e.g. Dataset, model, etc), "
                    f"consider passing a dictionary that will be populated with the instantiated "
                    f"objects so they can be reused."
                )
            else:
                _instantiated_objects_cache[path_so_far] = instantiated_obj

            if not hasattr(instantiated_obj, attr_part):
                # _instantiated_objects_cache[path_so_far] = instantiated_obj
                break

            obj = getattr(instantiated_obj, attr_part)
        else:
            # Found the attribute!
            return obj
    raise RuntimeError(
        f"Could not find any of these attributes {attributes} from the instantiated objects: "
        + str({k: type(v) for k, v in _instantiated_objects_cache.items()})
        # + "\n".join([f"- {k}: {type(v)}" for k, v in all_init_field_items.items()])
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

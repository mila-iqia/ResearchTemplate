from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import MISSING, field, fields, is_dataclass
from logging import getLogger as get_logger
from typing import (
    Any,
    Literal,
    TypeVar,
)

import hydra_zen
from hydra_zen.third_party.pydantic import pydantic_parser
from hydra_zen.typing._implementations import Partial as _Partial
from omegaconf import DictConfig, OmegaConf

if typing.TYPE_CHECKING:
    from project.configs.config import Config

logger = get_logger(__name__)

T = TypeVar("T")


def interpolate_config_attribute(*attributes: str, default: Any | Literal[MISSING] = MISSING):
    """Use this in a config to to get an attribute from another config after it is instantiated.

    Multiple attributes can be specified, which will lead to trying each of them in order until the
    attribute is found. If none are found, then an error will be raised.

    For example, if we only know the number of classes in the datamodule after it is instantiated,
    we can set this in the network config so it is created with the right number of output dims.

    ```yaml
    _target_: torchvision.models.resnet50
    num_classes: ${instance_attr:datamodule.num_classes}
    ```

    This is equivalent to:

    >>> import hydra_zen
    >>> import torchvision.models
    >>> resnet50_config = hydra_zen.builds(
    ...     torchvision.models.resnet50,
    ...     num_classes=interpolate_config_attribute("datamodule.num_classes"),
    ...     populate_full_signature=True,
    ... )
    >>> print(hydra_zen.to_yaml(resnet50_config))  # doctest: +NORMALIZE_WHITESPACE
    _target_: torchvision.models.resnet.resnet50
    weights: null
    progress: true
    num_classes: ${instance_attr:datamodule.num_classes}
    """
    if default is MISSING:
        return "${instance_attr:" + ",".join(attributes) + "}"
    return "${instance_attr:" + ",".join(attributes) + ":" + str(default) + "}"


def interpolated_field(
    interpolation: str,
    default: T | Literal[MISSING] = MISSING,
    default_factory: Callable[[], T] | Literal[MISSING] = MISSING,
    instance_attr: bool = False,
) -> T:
    """Field with a default value computed with a OmegaConf-style interpolation when appropriate.

    When the dataclass is created by Hydra / OmegaConf, the interpolation is used.
    Otherwise, behaves as usual (either using default or calling the default_factory).

    Parameters
    ----------
    interpolation: The string interpolation to use to get the default value.
    default: The default value to use when not in a hydra/OmegaConf context.
    default_factory: The default value to use when not in a hydra/OmegaConf context.
    instance_attr: Whether to use the `instance_attr` custom resolver to run the interpolation \
        with respect to instantiated objects instead of their configs.
        Passing `interpolation='${instance_attr:some_config.some_attr}'` has the same effect.

    This last parameter is important, since in order to retrieve the instance attribute, we need to
    instantiate the objects, which could be expensive. These instantiated objects are reused at
    least, but still, be mindful when using this parameter.
    """
    assert "${" in interpolation and "}" in interpolation

    if instance_attr:
        if not interpolation.startswith("${instance_attr:"):
            interpolation = interpolation.removeprefix("${")
            interpolation = "${instance_attr:" + interpolation

    if default is MISSING and default_factory is MISSING:
        raise RuntimeError(
            "Interpolated fields currently still require a default value or default factory for "
            "when they are used outside the Hydra/OmegaConf context."
        )
    return field(
        default_factory=functools.partial(
            _default_factory,
            interpolation=interpolation,
            default=default,
            default_factory=default_factory,
        )
    )


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


def get_full_name(object_type: type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_outer_class(inner_class: type) -> type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


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


def register_instance_attr_resolver(instantiated_objects_cache: dict[str, Any]) -> None:
    OmegaConf.register_new_resolver(
        "instance_attr",
        functools.partial(
            get_instantiated_attr,
            _instantiated_objects_cache=instantiated_objects_cache,
        ),
        replace=True,
    )


def resolve_dictconfig(dict_config: DictConfig) -> Config:
    # Important: Register this fancy little resolver here so we can get attributes of the
    # instantiated objects, not just the configs!
    instantiated_objects_cache: dict[str, Any] = {}
    register_instance_attr_resolver(instantiated_objects_cache)
    # Convert the "raw" DictConfig (which uses the `Config` class to define it's structure)
    # into an actual `Config` object:
    config = OmegaConf.to_object(dict_config)
    from project.configs.config import Config

    assert isinstance(config, Config)
    # If we had to instantiate some of the configs into objects in order to find the interpolated
    # values (e.g. ${instance_attr:datamodule.dims} or similar in order to construct the network),
    # then we don't waste that, put the object instance into the config.
    # TODO: This isn't quite correct typing-wise, since for example the field for `datamodule` is a
    # `DataModuleConfig` while we're setting it to a value of `LightningDataModule` if we
    # instantiated it.
    # TODO: We don't actually have LightningDataModule objects here! We only have these
    # `<DatamoduleName>Config` objects, so we can't get the attributes like `dims` properly!
    for attribute, pre_instantiated_object in instantiated_objects_cache.items():
        if not has_attr(config, attribute):
            logger.debug(
                f"Leftover temporarily-instantiated attribute {attribute} in the instantiated "
                f"objects cache."
            )
            continue
        value_in_config = get_attr(config, attribute)
        if pre_instantiated_object != value_in_config:
            logger.debug(
                f"Overwriting the config at {attribute} with the pre-instantiated "
                f"object {pre_instantiated_object}"
            )
            set_attr(config, attribute, pre_instantiated_object)

    return config


def get_instantiated_attr(
    *attributes: str,
    _instantiated_objects_cache: MutableMapping[str, Any] | None = None,
):
    """Quite hacky: Allows interpolations to get the value of the objects, rather than configs."""
    if not attributes:
        raise RuntimeError("Need to pass one or more attributes to this resolver.")
    assert being_called_in_hydra_context()
    logger.debug(f"Custom resolver is being called to get the value of {attributes}.")

    current_frame = inspect.currentframe()
    assert current_frame
    assert current_frame.f_back
    frame_infos = inspect.getouterframes(current_frame.f_back)

    # QUITE HACKY: These local variables are defined in the _to_object function inside OmegaConf.

    init_field_items: list[dict[str, Any]] = []
    non_init_field_items: list[dict[str, Any]] = []

    for frame_info in frame_infos:
        if frame_info.function == DictConfig._to_object.__name__:
            _self_obj: DictConfig = frame_info.frame.f_locals["self"]

            assert "init_field_items" in frame_info.frame.f_locals
            frame_init_field_items = frame_info.frame.f_locals["init_field_items"]
            logger.debug(
                f"Adding the following items into the init field items: {frame_init_field_items}"
            )
            init_field_items.append(frame_init_field_items.copy())

            assert "non_init_field_items" in frame_info.frame.f_locals
            frame_non_init_field_items = frame_info.frame.f_locals["non_init_field_items"]
            non_init_field_items.append(frame_non_init_field_items.copy())
            logger.debug(
                f"Adding the following items into the init field items: {frame_non_init_field_items}"
            )
    assert init_field_items or non_init_field_items
    # Okay so we now have all the *instantiated* attributes! We can do the interpolation by
    # getting its value!
    # BUG: This isn't quite working as I'd like, the interpolation is trying to get the attribute
    # on the object *instance*, but this is getting called during the first OmegaConf.to_object
    # which is only responsible for creating the configs, not the objects!
    all_init_field_items = ChainMap(*reversed(init_field_items))

    # BUG: There are some unexpected attributes being stored in the instantiated objects cache
    # which are fields of the dataclass (e.g. `layer_widths, sigma_gen, batch_size`, etc instead of
    # being configs!
    if _instantiated_objects_cache is not None:
        _instantiated_objects_cache.update(all_init_field_items)

    objects_cache: Mapping[str, Any] = _instantiated_objects_cache or all_init_field_items

    for attribute in attributes:
        logger.debug(f"Looking for instantiated attribute {attribute} from {all_init_field_items}")
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
                (isinstance(obj, dict | DictConfig) and "_target_" in obj)
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
                logger.error(
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
                    f"objects so they can be reused instead of being discarded."
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
        + str({k: type(v) for k, v in objects_cache.items()})
        # + "\n".join([f"- {k}: {type(v)}" for k, v in all_init_field_items.items()])
    )


def being_called_in_hydra_context() -> bool:
    import hydra.core.utils
    import omegaconf._utils
    import omegaconf.base

    return _being_called_by(
        omegaconf._utils.get_dataclass_data,
        omegaconf.base.Container._resolve_interpolation_from_parse_tree,
        hydra.core.utils.run_job,
    )


def _being_called_by(*functions: Callable) -> bool:
    current = inspect.currentframe()
    assert current

    caller_frames = inspect.getouterframes(current, context=0)
    if any(
        frame_info.function in [function.__name__ for function in functions]
        for frame_info in caller_frames
    ):
        return True
    return False


def _default_factory(
    interpolation: str,
    default: T | Literal[dataclasses.MISSING] = dataclasses.MISSING,
    default_factory: Callable[[], T] | Literal[dataclasses.MISSING] = dataclasses.MISSING,
) -> T:
    if being_called_in_hydra_context():
        return interpolation  # type: ignore
    if default_factory is not dataclasses.MISSING:
        return default_factory()
    return default  # type: ignore


instantiate = functools.partial(hydra_zen.instantiate, _target_wrapper_=pydantic_parser)

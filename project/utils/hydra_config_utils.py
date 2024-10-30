import importlib
import inspect
import typing
from collections.abc import Callable
from logging import getLogger as get_logger

import hydra_zen
from hydra.core.config_store import ConfigStore

from project.utils.hydra_utils import get_outer_class

logger = get_logger(__name__)


def get_config_loader():
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra._internal.utils import create_automatic_config_search_path

    from project.main import PROJECT_NAME

    search_path = create_automatic_config_search_path(
        calling_file=None, calling_module=None, config_path=f"pkg://{PROJECT_NAME}.configs"
    )
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    return config_loader


def get_all_configs_in_group(group_name: str) -> list[str]:
    # note: here we're copying a bit of the internal code from Hydra so that we also get the
    # configs that are just yaml files, in addition to the configs we added programmatically to the
    # configstores.

    # names_yaml = cs.list(group_name)
    # names = [name.rpartition(".")[0] for name in names_yaml]
    # if "base" in names:
    #     names.remove("base")
    # return names

    return get_config_loader().get_group_options(group_name)


def get_target_of_config(
    config_group: str, config_name: str, _cs: ConfigStore | None = None
) -> Callable:
    """Returns the class that is to be instantiated by the given config name.

    In the case of inner dataclasses (e.g. Model.HParams), this returns the outer class (Model).
    """
    # TODO: Rework, use the same mechanism as in auto-schema.py
    if _cs is None:
        from project.configs import cs as _cs

    config_loader = get_config_loader()
    _, caching_repo = config_loader._parse_overrides_and_create_caching_repo(
        config_name=None, overrides=[]
    )
    # todo: support both `.yml` and `.yaml` extensions for config files.
    for extension in ["yaml", "yml"]:
        config_result = caching_repo.load_config(f"{config_group}/{config_name}.{extension}")
        if config_result is None:
            continue
        try:
            return hydra_zen.get_target(config_result.config)  # type: ignore
        except TypeError:
            pass
    from hydra.plugins.config_source import ConfigLoadError

    try:
        config_node = _cs._load(f"{config_group}/{config_name}.yaml")
    except ConfigLoadError as error_yaml:
        try:
            config_node = _cs._load(f"{config_group}/{config_name}.yml")
        except ConfigLoadError:
            raise ConfigLoadError(
                f"Unable to find a config {config_group}/{config_name}.yaml or {config_group}/{config_name}.yml!"
            ) from error_yaml

    if "_target_" in config_node.node:
        # BUG: This won't work for nested classes! "module.class.class"
        target: str = config_node.node["_target_"]
        return import_object(target)
        # module_name, _, class_name = target.rpartition(".")
        # module = importlib.import_module(module_name)
        # target = getattr(module, class_name)
        # return target

    # If it doesn't have a target, then assume that it's an inner dataclass like this:
    """
    ```python
    class Model:
        class HParams:
            ...
        def __init__(self, ...): # (with an arg of type HParams)
            ...
    """
    # NOTE: A bit hacky, might break.
    hparam_type = config_node.node._metadata.object_type
    target_type = get_outer_class(hparam_type)
    return target_type


def import_object(target_path: str):
    """Imports the object at the given path.

    ## Examples

    ```python
    assert False
    ```
    """
    assert not target_path.endswith(
        ".py"
    ), "expect a valid python path like 'module.submodule.object'"
    if "." not in target_path:
        return importlib.import_module(target_path)

    parts = target_path.split(".")
    try:
        return importlib.import_module(name=f".{parts[-1]}", package=".".join(parts[:-1]))
    except (ModuleNotFoundError, AttributeError):
        pass

    for i in range(1, len(parts)):
        module_name = ".".join(parts[:i])
        obj_path = parts[i:]
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_path[0])
            for part in obj_path[1:]:
                obj = getattr(obj, part)
            return obj
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ModuleNotFoundError(f"Unable to import the {target_path=}!")


def get_all_configs_in_group_of_type(
    config_group: str,
    config_target_type: type | tuple[type, ...],
    include_subclasses: bool = True,
    excluding: type | tuple[type, ...] = (),
) -> list[str]:
    """Returns the names of all the configs in the given config group that have this target or a
    subclass of it."""
    config_names = get_all_configs_in_group(config_group)
    names_to_targets = {
        config_name: get_target_of_config(config_group, config_name)
        for config_name in config_names
    }

    names_to_types: dict[str, type] = {}
    for name, target in names_to_targets.items():
        if inspect.isclass(target):
            names_to_types[name] = target
            continue

        if (
            (inspect.isfunction(target) or inspect.ismethod(target))
            and (annotations := typing.get_type_hints(target))
            and (return_type := annotations.get("return"))
            and (inspect.isclass(return_type) or inspect.isclass(typing.get_origin(return_type)))
        ):
            # Resolve generic aliases if present.
            return_type = typing.get_origin(return_type) or return_type
            logger.info(
                f"Assuming that the function {target} creates objects of type {return_type} based "
                f"on its return type annotation."
            )
            names_to_types[name] = return_type
            continue

        logger.warning(
            RuntimeWarning(
                f"Unable to tell what kind of object will be created by the target {target} of "
                f"config {name} in group {config_group}. This config will be skipped in tests."
            )
        )
    config_target_type = (
        config_target_type if isinstance(config_target_type, tuple) else (config_target_type,)
    )
    if excluding is not None:
        exclude = (excluding,) if isinstance(excluding, type) else excluding
        names_to_types = {
            name: object_type
            for name, object_type in names_to_types.items()
            if (
                not issubclass(object_type, exclude)
                if include_subclasses
                else object_type not in exclude
            )
        }

    return [
        name
        for name, object_type in names_to_types.items()
        if (
            issubclass(object_type, config_target_type)
            if include_subclasses
            else object_type in config_target_type
        )
    ]


def get_all_configs_in_group_with_target(group_name: str, some_type: type) -> list[str]:
    """Retrieves the names of all the configs in the given group that are used to construct objects
    of the given type."""
    config_names = get_all_configs_in_group(group_name)
    names_to_target = {
        config_name: get_target_of_config(group_name, config_name) for config_name in config_names
    }
    return [name for name, object_type in names_to_target.items() if object_type == some_type]

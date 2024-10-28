"""Scripts that creates Schemas for hydra config files.

This is very helpful when using Hydra! It shows the user what options are available, along with their
description and default values, and displays errors if you have config files with invalid values.

## todos
- [ ] Support '???' as a value for any property.
- [ ] Modify the schema to support omegaconf directives like ${oc.env:VAR_NAME} and our custom directives like ${instance_attr} and so on.
- [ ] todo: Make a hydra plugin that creates the schemas for configs when hydra is loading stuff.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import importlib
import inspect
import json
import logging
import os.path
import subprocess
import sys
import time
import typing
import warnings
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict, TypeVar

import docstring_parser as dp
import flax
import flax.linen
import flax.struct
import hydra.errors
import hydra.plugins
import hydra.utils
import hydra_zen
import lightning.pytorch.callbacks
import omegaconf
import pydantic
import pydantic.schema
import rich.logging
import tqdm
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from tqdm.rich import tqdm_rich
from typing_extensions import NotRequired, Required
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,  # noqa
    PatternMatchingEventHandler,  # noqa
)
from watchdog.observers import Observer

logger = get_logger(__name__)

_SpecialEntries = TypedDict(
    "_SpecialEntries", {"$defs": dict[str, dict], "$schema": str}, total=False
)


def register_auto_schema_plugin():
    import hydra.core.plugins

    hydra.core.plugins.Plugins.instance().register(AutoSchemaPlugin)


class PropertySchema(_SpecialEntries, total=False):
    title: str
    type: str
    description: str
    default: Any
    examples: list[str]
    deprecated: bool
    readOnly: bool
    writeOnly: bool
    const: Any
    enum: list[Any]


# S = TypeVar("S", bound=PropertySchema, covariant=True)


class OneOf(TypedDict):
    oneOf: Sequence[PropertySchema | ArrayPropertySchema | ObjectSchema | StringPropertySchema]


class ArrayPropertySchema(PropertySchema, total=False):
    type: Literal["array"]
    items: Required[PropertySchema | OneOf]
    minItems: int
    maxItems: int
    uniqueItems: bool


class StringPropertySchema(PropertySchema, total=False):
    type: Literal["string"]
    pattern: str


class _PropertyNames(TypedDict):
    pattern: str


class ObjectSchema(PropertySchema, total=False):
    type: Literal["object"]
    # annoying that we have to include the subclasses here!
    properties: MutableMapping[
        str, PropertySchema | ArrayPropertySchema | StringPropertySchema | ObjectSchema
    ]
    patternProperties: Mapping[str, PropertySchema | StringPropertySchema]
    propertyNames: _PropertyNames
    minProperties: int
    maxProperties: int


class Schema(TypedDict, total=False):
    # "$defs":
    title: str
    description: str
    type: str

    # annoying that we have to include the subclasses here!
    properties: Required[
        MutableMapping[
            str, PropertySchema | ArrayPropertySchema | StringPropertySchema | ObjectSchema
        ]
    ]
    required: NotRequired[list[str]]

    additionalProperties: NotRequired[bool]

    dependentRequired: NotRequired[MutableMapping[str, list[str]]]
    """ https://json-schema.org/understanding-json-schema/reference/conditionals#dependentRequired """


HYDRA_CONFIG_SCHEMA = Schema(
    title="Default Schema for any Hydra config file.",
    description="Schema created by the `auto_schema.py` script.",
    properties={
        "defaults": ArrayPropertySchema(
            title="Hydra defaults",
            description="Hydra defaults for this config. See https://hydra.cc/docs/advanced/defaults_list/",
            type="array",
            items=OneOf(
                oneOf=[
                    ObjectSchema(
                        type="object",
                        propertyNames={"pattern": r"^(override\s*)?(/?\w*)+$"},
                        patternProperties={
                            # todo: support package directives with @?
                            # todo: Create a enum schema using the available options in the config group!
                            # override network: something  -> `something` value should be in the available options for network.
                            r"^(override\s*)?(/?\w*)*$": StringPropertySchema(
                                type="string", pattern=r"\w*(.yaml|.yml)?$"
                            ),
                        },
                        minProperties=1,
                        maxProperties=1,
                    ),
                    StringPropertySchema(type="string", pattern=r"^\w+(.yaml|.yml)?$"),
                    ObjectSchema(
                        type="object",
                        propertyNames=_PropertyNames(pattern=r"^(override\s*)?(/?\w*)+$"),
                        patternProperties={
                            r"^(override\s*)?(/?\w*)*$": PropertySchema(type="null"),
                        },
                        minProperties=1,
                        maxProperties=1,
                    ),
                ],
            ),
            uniqueItems=True,
        ),
        "_target_": StringPropertySchema(
            type="string",
            title="Target",
            description="Target to instantiate.\nSee https://hydra.cc/docs/advanced/instantiate_objects/overview/",
        ),
        "_convert_": StringPropertySchema(
            type="string",
            enum=["none", "partial", "object", "all"],
            title="Convert",
            description="See https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies",
        ),
        "_partial_": PropertySchema(
            type="boolean",
            title="Partial",
            description=(
                "Whether this config calls the target function when instantiated, or creates "
                "a `functools.partial` that will call the target.\n"
                "See: https://hydra.cc/docs/advanced/instantiate_objects/overview"
            ),
        ),
        "_recursive_": PropertySchema(
            type="boolean",
            title="Recursive",
            description=(
                "Whether instantiating this config should recursively instantiate children configs.\n"
                "See: https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation"
            ),
        ),
    },
    dependentRequired={
        "_convert_": ["_target_"],
        "_partial_": ["_target_"],
        "_args_": ["_target_"],
        "_recursive_": ["_target_"],
    },
)


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.ERROR,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    parser = argparse.ArgumentParser()

    # TODO: Fix this

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help=(
            "The directory containing hydra config files for which schemas should be generated. "
            "If unset, this will default to the first directory called 'configs' in the repo root."
        ),
    )
    parser.add_argument("--schemas-dir", type=Path, default=Path.cwd() / ".schemas")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--regen-schemas", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction)
    parser.add_argument("--watch", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--add-headers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Always add headers to the yaml config files, instead of the default "
            "behaviour which is to first try to add an entry in the vscode "
            "settings.json file."
        ),
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-q", "--quiet", dest="quiet", action=argparse.BooleanOptionalAction
    )
    verbosity_group.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    configs_dir: Path | None = args.configs_dir
    schemas_dir: Path = args.schemas_dir
    repo_root: Path = args.repo_root
    regen_schemas: bool = args.regen_schemas
    stop_on_error: bool = args.stop_on_error
    quiet: bool = args.quiet
    verbose: int = args.verbose
    add_headers: bool = args.add_headers
    watch: bool = args.watch

    if configs_dir is None:
        configs_dir = next(repo_root.rglob("configs"), None)
    if configs_dir is None:
        raise ValueError(
            f"--configs-dir was not passed, and no hydra configs directory was found in "
            f"the {repo_root=}!"
        )

    if quiet:
        logger.setLevel(logging.NOTSET)
    elif verbose:
        if verbose >= 3:
            logger.setLevel(logging.DEBUG)
        elif verbose == 2:
            logger.setLevel(logging.INFO)
        else:
            assert verbose == 1
            logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    if watch:
        observer = Observer()
        handler = AutoSchemaHandler(
            repo_root=repo_root,
            configs_dir=configs_dir,
            schemas_dir=schemas_dir,
            regen_schemas=regen_schemas,
            stop_on_error=stop_on_error,
            quiet=quiet,
            add_headers=add_headers,
        )
        observer.schedule(handler, str(configs_dir), recursive=True)
        observer.start()
        logger.info("Watching for changes in the config files.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        return

    add_schemas_to_all_hydra_configs(
        repo_root=repo_root,
        configs_dir=configs_dir,
        schemas_dir=schemas_dir,
        regen_schemas=regen_schemas,
        stop_on_error=stop_on_error,
        quiet=quiet,
        add_headers=add_headers,
    )
    logger.info("Done updating the schemas for the Hydra config files.")


def _yaml_files_in(configs_dir):
    return list(configs_dir.rglob("*.yaml")) + list(configs_dir.rglob("*.yml"))


def add_schemas_to_all_hydra_configs(
    repo_root: Path,
    configs_dir: Path,
    schemas_dir: Path | None = None,
    regen_schemas: bool = False,
    stop_on_error: bool = False,
    quiet: bool = False,
    add_headers: bool | None = False,
):
    """Adds schemas to all the passed Hydra config files.

    Parameters:
        config_files: List of paths to the Hydra config files. If None, all YAML files in configs_dir are used.
        repo_root: The root directory of the repository.
        configs_dir: The directory containing the Hydra config files.
        schemas_dir: The directory to store the generated schema files. Defaults to ".schemas" in the repo_root.
        regen_schemas: If True, regenerate schemas even if they already exist. Defaults to False.
        stop_on_error: If True, raise an exception on error. Defaults to False.
        quiet: If True, suppress progress bar output. Defaults to False.
        add_headers: Determines how to associate schema files with config files.

            - If None, try adding to VSCode settings first, then fallback to adding headers.
            - If False, only use VSCode settings.
            - If True, only add headers.
    """

    config_files = _yaml_files_in(configs_dir)
    if not config_files:
        warnings.warn(RuntimeWarning("No config files were passed. Skipping."))
        return

    if schemas_dir is None:
        schemas_dir = repo_root / ".schemas"

    if schemas_dir.is_relative_to(repo_root):
        _add_schemas_dir_to_gitignore(schemas_dir, repo_root=repo_root)

    config_file_to_schema_file: dict[Path, Path] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
        pbar = tqdm_rich(
            config_files,
            desc="Creating schemas for Hydra config files...",
            total=len(config_files),
            leave=False,
            disable=quiet,
        )

    for config_file in pbar:
        pretty_config_file_name = config_file.relative_to(configs_dir)
        schema_file = _get_schema_file_path(config_file, schemas_dir)

        # todo: check the modification time. If the config file was modified after the schema file, regen the schema file.

        if schema_file.exists():
            schema_file_modified_time = datetime.datetime.fromtimestamp(
                schema_file.stat().st_mtime
            )
            config_file_modified_time = datetime.datetime.fromtimestamp(
                config_file.stat().st_mtime
            )
            # Add a delay to account for the time it takes to modify the the config file at the end to remove a header.
            if (config_file_modified_time - schema_file_modified_time) > datetime.timedelta(
                seconds=10
            ):
                logger.info(
                    f"Config file {pretty_config_file_name} was modified, regenerating the schema."
                )
            elif regen_schemas:
                pass  # regenerate it.
            elif _is_incomplete_schema(schema_file):
                logger.info(
                    f"Unable to properly create the schema for {pretty_config_file_name} last time. Trying again."
                )
            else:
                logger.debug(
                    f"Schema file {_relative_to_cwd(schema_file)} was already successfully created. Skipping."
                )
                continue

        pbar.set_postfix_str(f"Creating schema for {pretty_config_file_name}")

        try:
            logger.debug(f"Creating a schema for {pretty_config_file_name}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                config = _load_config(config_file, configs_dir=configs_dir, repo_root=repo_root)
            schema = create_schema_for_config(
                config, config_file=config_file, configs_dir=configs_dir, repo_root=repo_root
            )
            schema_file.parent.mkdir(exist_ok=True, parents=True)
            schema_file.write_text(json.dumps(schema, indent=2).rstrip() + "\n\n")
            _set_is_incomplete_schema(schema_file, False)
        except (
            pydantic.errors.PydanticSchemaGenerationError,
            hydra.errors.MissingConfigException,
            hydra.errors.ConfigCompositionException,
            omegaconf.errors.InterpolationResolutionError,
            Exception,  # todo: remove this to harden the code.
        ) as exc:
            logger.warning(
                RuntimeWarning(
                    f"Unable to create a schema for config {pretty_config_file_name}: {exc}"
                )
            )
            if stop_on_error:
                raise

            schema = copy.deepcopy(HYDRA_CONFIG_SCHEMA)
            schema["additionalProperties"] = True
            schema["title"] = f"Partial schema for {pretty_config_file_name}"
            schema["description"] = (
                f"(errors occurred while trying to create the schema from the signature:\n{exc}"
            )
            schema_file.write_text(json.dumps(schema, indent=2) + "\n")
            _set_is_incomplete_schema(schema_file, True)

        config_file_to_schema_file[config_file] = schema_file

    # Option 1: Add a vscode setting that associates the schema file with the yaml files. (less intrusive perhaps).
    # Option 2: Add a header to the yaml files that points to the schema file.

    # If add_headers is None, try option 1, then fallback to option 2.
    # If add_headers is False, only use option 1
    # If add_headers is True, only use option 2

    if not add_headers:
        try:
            _install_yaml_vscode_extension()
        except OSError:
            pass

        try:
            _add_schemas_to_vscode_settings(config_file_to_schema_file, repo_root=repo_root)
        except Exception as exc:
            logger.error(
                f"Unable to write schemas in the vscode settings file. "
                f"Falling back to adding a header to config files. (exc={exc})"
            )
            if add_headers is not None:
                # Unable to do it. Don't try to add headers, just return.
                return
        else:
            # Success. Return.
            return

    logger.debug("Adding headers to config files to point to the schemas to use.")
    for config_file, schema_file in config_file_to_schema_file.items():
        add_schema_header(config_file, schema_path=schema_file)


def _add_schemas_dir_to_gitignore(schemas_dir: Path, repo_root: Path):
    _rel = schemas_dir.relative_to(repo_root)
    _gitignore_file = repo_root / ".gitignore"
    if not any(line.startswith(str(_rel)) for line in _gitignore_file.read_text().splitlines()):
        logger.info(f"Adding entry in .gitignore for the schemas directory ({schemas_dir})")
        with _gitignore_file.open("a") as f:
            f.write(f"{_rel}\n")


_incomplete_schema_xattr = "user.schema_error"


def _is_incomplete_schema(schema_file: Path) -> bool:
    try:
        return os.getxattr(schema_file, _incomplete_schema_xattr) == bytes(1)
    except OSError:
        return False


def _set_is_incomplete_schema(schema_file: Path, val: bool):
    os.setxattr(schema_file, _incomplete_schema_xattr, bytes(val))


def _relative_to_cwd(p: str | Path):
    return Path(p).relative_to(Path.cwd())


def _install_yaml_vscode_extension():
    logger.debug(
        "Running `code --install-extension redhat.vscode-yaml` to install the yaml extension for vscode."
    )
    output = subprocess.check_output(
        ("code", "--install-extension", "redhat.vscode-yaml"), text=True
    )
    logger.debug(output)


def _read_json(file: Path) -> dict:
    file_contents = file.read_text()
    # Remove any trailing commas from the content:
    file_contents = (
        # Remove any trailing "," that would make it invalid JSON.
        "".join(file_contents.split()).replace(",}", "}").replace(",]", "]")
    )
    return json.loads(file_contents)


def _add_schemas_to_vscode_settings(
    config_file_to_schema_file: dict[Path, Path],
    repo_root: Path,
) -> None:
    # Make the vscode settings file if necessary:
    vscode_dir = repo_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True, parents=False)
    vscode_settings_file = vscode_dir / "settings.json"
    vscode_settings_file.touch(exist_ok=True)

    # TODO: Use Jsonc to load the file and preserve comments?
    logger.debug(f"Reading the VsCode settings file at {vscode_settings_file}.")
    vscode_settings: dict[str, Any] = _read_json(vscode_settings_file)

    # logger.debug(f"Vscode settings: {vscode_settings}")
    # Avoid the popup and do users a favour by disabling telemetry.
    vscode_settings.setdefault("redhat.telemetry.enabled", False)

    # TODO: Should probably overwrite the schemas entry if we're passed the --regen-schemas flag,
    # since otherwise we might accumulate schemas for configs that aren't there anymore.
    yaml_schemas_setting: dict[str, str | list[str]] = vscode_settings.setdefault(
        "yaml.schemas", {}
    )

    # Write all the schemas
    for config_file, schema_file in config_file_to_schema_file.items():
        assert schema_file.exists()

        _root = vscode_settings_file.parent.parent
        schema_key = str(schema_file.relative_to(_root))
        path_to_add = str(config_file.absolute())

        if schema_key not in yaml_schemas_setting:
            yaml_schemas_setting[schema_key] = path_to_add
        elif isinstance(files_associated_with_schema := yaml_schemas_setting[schema_key], str):
            files = sorted(set([files_associated_with_schema, path_to_add]))
            yaml_schemas_setting[schema_key] = files[0] if len(files) == 1 else files
        else:
            files = sorted(set(files_associated_with_schema + [path_to_add]))
            yaml_schemas_setting[schema_key] = files[0] if len(files) == 1 else files

    vscode_settings_file.write_text(json.dumps(vscode_settings, indent=2))
    logger.debug(
        f"Updated the yaml schemas in the vscode settings file at {vscode_settings_file}."
    )

    # If this worked, then remove any schema directives from the config files.
    for config_file, schema_file in config_file_to_schema_file.items():
        assert schema_file.exists()
        config_lines = config_file.read_text().splitlines()
        lines_to_remove: list[int] = []
        for i, line in enumerate(config_lines):
            if line.strip().startswith("# yaml-language-server: $schema="):
                lines_to_remove.append(i)
        for line_to_remove in reversed(lines_to_remove):
            config_lines.pop(line_to_remove)
        config_file.write_text("\n".join(config_lines).rstrip() + ("\n" if config_lines else ""))


def _get_schema_file_path(config_file: Path, schemas_dir: Path):
    config_group = config_file.parent
    schema_file = schemas_dir / f"{config_group.name}_{config_file.stem}_schema.json"
    return schema_file


def _all_subentries_with_target(config: dict) -> dict[tuple[str, ...], dict]:
    """Iterator that yields all the nested config entries that have a _target_."""
    entries = {}
    if "_target_" in config:
        entries[()] = config
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in _all_subentries_with_target(value).items():
                entries[(key, *subkey)] = subvalue
    return entries


def create_schema_for_config(
    config: dict | DictConfig, config_file: Path, configs_dir: Path | None, repo_root: Path
) -> Schema | ObjectSchema:
    """IDEA: Create a schema for the given config.

    - If you encounter a key, add it to the schema.
    - If you encounter a value with a _target_, use a dedicated function to get the schema for that target, and merge it into the current schema.
    - Only the top-level config (`config`) can have a `defaults: list[str]` key.
        - Should ideally load the defaults and merge this schema on top of them.
    """

    _config_dict = (
        OmegaConf.to_container(config, resolve=False) if isinstance(config, DictConfig) else config
    )
    assert isinstance(_config_dict, dict)

    schema = copy.deepcopy(HYDRA_CONFIG_SCHEMA)
    pretty_path = config_file.relative_to(configs_dir) if configs_dir else config_file
    schema["title"] = f"Auto-generated schema for {pretty_path}"

    if config_file.exists() and configs_dir:
        # note: the `defaults` list gets consumed by Hydra in `_load_config`, so we actually re-read the
        # config file to get the `defaults`, if present.
        # TODO: There still defaults here, only during unit tests, even though _load_config should have consumed them!
        if "defaults" in config:
            schema = _update_schema_from_defaults(
                config_file=config_file,
                schema=schema,
                defaults=config["defaults"],
                configs_dir=configs_dir,
                repo_root=repo_root,
            )

    # Config file that contains entries that may or may not have a _target_.
    schema["additionalProperties"] = "_target_" not in config

    for keys, value in _all_subentries_with_target(_config_dict).items():
        is_top_level: bool = not keys

        # logger.debug(f"Handling key {'.'.join(keys)} in config at path {config_file}")

        nested_value_schema = _get_schema_from_target(value)

        if "$defs" in nested_value_schema:
            # note: can't have a $defs key in the schema.
            schema.setdefault("$defs", {}).update(  # type: ignore
                nested_value_schema.pop("$defs")
            )
            assert "properties" in nested_value_schema

        if is_top_level:
            schema = _merge_dicts(schema, nested_value_schema, conflict_handler=overwrite)
            continue

        parent_keys, last_key = keys[:-1], keys[-1]
        where_to_set: Schema | ObjectSchema = schema
        for key in parent_keys:
            where_to_set = where_to_set.setdefault("properties", {}).setdefault(
                key, {"type": "object"}
            )  # type: ignore
            if "_target_" not in where_to_set:
                where_to_set["additionalProperties"] = True

        # logger.debug(f"Using schema from nested value at keys {keys}: {nested_value_schema}")

        if "properties" not in where_to_set:
            where_to_set["properties"] = {last_key: nested_value_schema}  # type: ignore
        elif last_key not in where_to_set["properties"]:
            assert isinstance(last_key, str)
            where_to_set["properties"][last_key] = nested_value_schema  # type: ignore
        else:
            where_to_set["properties"] = _merge_dicts(  # type: ignore
                where_to_set["properties"],
                {last_key: nested_value_schema},  # type: ignore
                conflict_handler=overwrite,
            )

    return schema


def _update_schema_from_defaults(
    config_file: Path,
    schema: Schema,
    defaults: list[str | dict[str, str]],
    configs_dir: Path,
    repo_root: Path,
):
    defaults_list = defaults

    for default in defaults_list:
        if default == "_self_":  # todo: does this actually make sense?
            continue
        # Note: The defaults can also have the .yaml or .yml extension, _load_config drops the
        # extension.
        if isinstance(default, str):
            assert not default.startswith("/")
            other_config_path = config_file.parent / default
        else:
            assert len(default) == 1
            key, val = next(iter(default.items()))
            other_config_path = config_file.parent / key / val
        logger.debug(f"Loading config of default {default}.")

        if not other_config_path.suffix:
            # If the other config file has the name without the extension, try both .yml and .yaml.
            for suffix in (".yml", ".yaml"):
                if other_config_path.with_suffix(suffix).exists():
                    other_config_path = other_config_path.with_suffix(suffix)
                    break

        # try:
        default_config = _load_config(
            other_config_path, configs_dir=configs_dir, repo_root=repo_root
        )
        # except omegaconf.errors.MissingMandatoryValue:
        #     default_config = OmegaConf.load(other_config_path)

        schema_of_default = create_schema_for_config(
            config=default_config,
            config_file=other_config_path,
            configs_dir=configs_dir,
            repo_root=repo_root,
        )

        logger.debug(f"Schema from default {default}: {schema_of_default}")
        logger.debug(f"Properties of {default=}: {list(schema_of_default['properties'].keys())}")  # type: ignore

        schema = _merge_dicts(  # type: ignore
            schema_of_default,  # type: ignore
            schema,  # type: ignore
            conflict_handlers={
                "_target_": overwrite,  # use the new target.
                "default": overwrite,  # use the new default?
                "title": overwrite,
                "description": overwrite,
            },
        )
        # todo: deal with this one here.
        if schema.get("additionalProperties") is False:
            schema.pop("additionalProperties")
    return schema


def overwrite(val_a: Any, val_b: Any) -> Any:
    return val_b


def keep_previous(val_a: Any, val_b: Any) -> Any:
    return val_a


conflict_handlers: dict[str, Callable[[Any, Any], Any]] = {}

_K = TypeVar("_K")
_V = TypeVar("_V")
_NestedDict = MutableMapping[_K, _V | "_NestedDict[_K, _V]"]

_D1 = TypeVar("_D1", bound=_NestedDict)
_D2 = TypeVar("_D2", bound=_NestedDict)


def _merge_dicts(
    a: _D1,
    b: _D2,
    path: list[str] = [],
    conflict_handlers: dict[str, Callable[[Any, Any], Any]] = conflict_handlers,
    conflict_handler: Callable[[Any, Any], Any] | None = None,
) -> _D1 | _D2:
    """Merge two nested dictionaries.

    >>> x = dict(b=1, c=dict(d=2, e=3))
    >>> y = dict(d=3, c=dict(z=2, f=4))
    >>> _merge_dicts(x, y)
    {'b': 1, 'c': {'d': 2, 'e': 3, 'z': 2, 'f': 4}, 'd': 3}
    >>> x
    {'b': 1, 'c': {'d': 2, 'e': 3}}
    >>> y
    {'d': 3, 'c': {'z': 2, 'f': 4}}
    """
    out = copy.deepcopy(a)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                out[key] = _merge_dicts(
                    a[key],
                    b[key],
                    path + [str(key)],
                    conflict_handlers={
                        k.removeprefix(f"{key}."): v for k, v in conflict_handlers.items()
                    },
                    conflict_handler=conflict_handler,
                )
            elif a[key] != b[key]:
                if specific_conflict_handler := conflict_handlers.get(key):
                    out[key] = specific_conflict_handler(a[key], b[key])  # type: ignore
                elif conflict_handler:
                    out[key] = conflict_handler(a[key], b[key])  # type: ignore

                # if any(key.split(".")[-1] == handler_name for  for prefix in ["_", "description", "title"]):
                #     out[key] = b[key]
                else:
                    raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            out[key] = copy.deepcopy(b[key])  # type: ignore
    return out


def _has_package_global_line(config_file: Path) -> int | None:
    """Returns whether the config file contains a `@package _global_` directive of hydra.

    See: https://hydra.cc/docs/advanced/overriding_packages/#overriding-the-package-via-the-package-directive
    """
    for line in config_file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        if line.removeprefix("#").strip().startswith("@package _global_"):
            return True
    return False


def _load_config(config_path: Path, configs_dir: Path, repo_root: Path) -> DictConfig:
    *config_groups, config_name = config_path.relative_to(configs_dir).with_suffix("").parts
    logger.debug(
        f"config_path: ./{_relative_to_cwd(config_path)}, {config_groups=}, {config_name=}, configs_dir: {configs_dir}"
    )
    config_group = "/".join(config_groups)

    from hydra.core.utils import setup_globals

    # todo: When would this normally be called?
    setup_globals()

    # FIXME!
    if configs_dir.is_relative_to(repo_root) and (configs_dir / "__init__.py").exists():
        config_module = str(configs_dir.relative_to(repo_root)).replace("/", ".")
        search_path = create_config_search_path(f"pkg://{config_module}")
    else:
        search_path = create_config_search_path(str(configs_dir))

    if _has_package_global_line(config_path):
        # Tricky: Here we load the global config but with the given config as an override.
        config_loader = ConfigLoaderImpl(config_search_path=search_path)
        top_config = config_loader.load_configuration(
            "config",  # todo: Fix this?
            overrides=[f"{config_group}={config_name}"],
            # todo: setting this here because it appears to be what's used in Hydra in a normal
            # run, even though RunMode.RUN would make more sense intuitively.
            run_mode=RunMode.MULTIRUN,
        )
        return top_config

    # Load the global config and get the node for the desired config.
    # TODO: Can this cause errors if configs in an unrelated subtree have required values?
    logger.debug(f"loading config {config_path}")
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        top_config = config_loader.load_configuration(
            f"{config_group}/{config_name}", overrides=[], run_mode=RunMode.MULTIRUN
        )
    # Retrieve the sub-entry in the config and return it.
    config = top_config
    for config_group in config_groups:
        config = config[config_group]
    return config


def add_schema_header(config_file: Path, schema_path: Path) -> None:
    """Add a comment in the yaml config file to tell yaml language server where to look for the
    schema.

    Importantly in the context of Hydra, this comment line should be added **after** any `#
    @package: <xyz>` directives of Hydra, otherwise Hydra doesn't use those directives properly
    anymore.
    """
    lines = config_file.read_text().splitlines(keepends=False)

    if config_file.parent is schema_path.parent:
        # TODO: Unsure when this branch would be used, and if it would differ.
        relative_path_to_schema = "./" + schema_path.name
    else:
        relative_path_to_schema = os.path.relpath(schema_path, start=config_file.parent)

    # Remove any existing schema lines.
    lines = [
        line for line in lines if not line.strip().startswith("# yaml-language-server: $schema=")
    ]

    # NOTE: This line can be placed anywhere in the file, not necessarily needs to be at the top,
    # and the yaml vscode extension will pick it up.
    new_line = f"# yaml-language-server: $schema={relative_path_to_schema}"

    package_global_line: int | None = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # BUG: IF the schema line comes before a @package: global comment, then the @package: _global_
        # comment is ignored by Hydra.
        # Locate the last package line (a bit unnecessary, since there should only be one).
        if line.startswith("#") and line.removeprefix("#").strip().startswith("@package"):
            package_global_line = i

    if package_global_line is None:
        # There's no package directive in the file.
        new_lines = [new_line, *lines]
    else:
        new_lines = lines.copy()
        # Insert the schema line after the package directive line.
        new_lines.insert(package_global_line + 1, new_line)

    result = "\n".join(new_lines).strip() + "\n"
    if config_file.read_text() != result:
        config_file.write_text(result)


def _get_schema_from_target(config: dict | DictConfig) -> ObjectSchema | Schema:
    assert isinstance(config, dict | DictConfig)
    # logger.debug(f"Config: {config}")
    target = hydra.utils.get_object(config["_target_"])

    if inspect.isclass(target) and issubclass(target, flax.linen.Module):
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_recursive=False,
            hydra_convert="all",
            zen_exclude=["parent"],
            zen_dataclass={"cls_name": target.__qualname__},
        )

    elif inspect.isclass(target) and issubclass(
        target, lightning.pytorch.callbacks.RichProgressBar
    ):
        from rich.style import Style  # noqa

        # todo: trying to fix this here.
        # RichProgressBarTheme.__annotations__[]
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_recursive=False,
            hydra_convert="all",
            zen_exclude=["theme"],
            zen_dataclass={"cls_name": target.__qualname__},
        )
    elif dataclasses.is_dataclass(target):
        # The target is a dataclass, so the schema is just the schema of the dataclass.
        object_type = target
    else:
        # The target is a type or callable.
        assert callable(target)
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_defaults=config.get("defaults", None),
            hydra_recursive=False,
            hydra_convert="all",
            zen_dataclass={"cls_name": target.__qualname__.replace(".", "_")},
            # zen_wrappers=pydantic_parser,  # unsure if this is how it works?
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            json_schema = pydantic.TypeAdapter(object_type).json_schema(
                mode="serialization",
                schema_generator=_MyGenerateJsonSchema,
                by_alias=False,
            )
            json_schema = typing.cast(Schema, json_schema)
        assert "properties" in json_schema
    except pydantic.PydanticSchemaGenerationError as e:
        raise NotImplementedError(f"Unable to get the schema with pydantic: {e}")

    assert "properties" in json_schema

    # Add a description
    json_schema["description"] = f"Based on the signature of {target}.\n" + json_schema.get(
        "description", ""
    )

    docs_to_search: list[dp.Docstring] = []

    if inspect.isclass(target):
        for target_or_base_class in inspect.getmro(target):
            if class_docstring := inspect.getdoc(target_or_base_class):
                docs_to_search.append(dp.parse(class_docstring))
            if init_docstring := inspect.getdoc(target_or_base_class.__init__):
                docs_to_search.append(dp.parse(init_docstring))
    else:
        assert inspect.isfunction(target) or inspect.ismethod(target), target
        docstring = inspect.getdoc(target)
        if docstring:
            docs_to_search = [dp.parse(docstring)]

    param_descriptions: dict[str, str] = {}
    for doc in docs_to_search:
        for param in doc.params:
            if param.description and param.arg_name not in param_descriptions:
                param_descriptions[param.arg_name] = param.description

    # Update the pydantic schema with descriptions:
    for property_name, property_dict in json_schema["properties"].items():
        if description := param_descriptions.get(property_name):
            property_dict["description"] = description
        else:
            property_dict["description"] = (
                f"The {property_name} parameter of the {target.__qualname__}."
            )

    if config.get("_partial_"):
        json_schema["required"] = []
    # Add some info on the target.
    if "_target_" not in json_schema["properties"]:
        json_schema["properties"]["_target_"] = {}
    else:
        assert isinstance(json_schema["properties"]["_target_"], dict)
    json_schema["properties"]["_target_"].update(
        PropertySchema(  # type: ignore
            type="string",
            title="Target",
            const=config["_target_"],
            # pattern=r"", # todo: Use a pattern to match python module import strings.
            description=(
                f"Target to instantiate, in this case: `{target}`\n"
                # f"* Source: <file://{relative_to_cwd(inspect.getfile(target))}>\n"
                # f"* Config file: <file://{config_file}>\n"
                f"See the Hydra docs for '_target_': https://hydra.cc/docs/advanced/instantiate_objects/overview/\n"
            ),
        )
    )

    # if the target takes **kwargs, then we don't restrict additional properties.
    json_schema["additionalProperties"] = inspect.getfullargspec(target).varkw is not None

    return json_schema


def _target_has_var_kwargs(config: DictConfig) -> bool:
    target = hydra_zen.get_target(config)  # type: ignore
    return inspect.getfullargspec(target).varkw is None


class _MyGenerateJsonSchema(GenerateJsonSchema):
    # def handle_invalid_for_json_schema(
    #     self, schema: core_schema.CoreSchema, error_info: str
    # ) -> JsonSchemaValue:
    #     raise PydanticOmit

    def enum_schema(self, schema: core_schema.EnumSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches an Enum value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        enum_type = schema["cls"]
        logger.debug(f"Enum of type {enum_type}")
        import torchvision.models.resnet

        if issubclass(enum_type, torchvision.models.WeightsEnum):

            @dataclasses.dataclass
            class Dummy:
                value: str

            slightly_changed_schema = schema | {
                "members": [Dummy(v.name) for v in schema["members"]]
            }
            return super().enum_schema(slightly_changed_schema)  # type: ignore
        return super().enum_schema(schema)


@hydra_zen.hydrated_dataclass(
    add_schemas_to_all_hydra_configs, populate_full_signature=True, zen_partial=True
)
class AutoSchemaPluginConfig:
    """Config for the AutoSchemaPlugin."""

    schemas_dir: Path | None = None
    regen_schemas: bool = False
    stop_on_error: bool = False
    quiet: bool = True
    add_headers: bool | None = False


config: AutoSchemaPluginConfig | None = None


class AutoSchemaPlugin(SearchPathPlugin):
    provider: str
    path: str
    _ALREADY_DID: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.config = config or AutoSchemaPluginConfig()
        self.fn = hydra_zen.instantiate(self.config)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if type(self)._ALREADY_DID:
            # TODO: figure out what's causing this.
            logger.debug(f"Avoiding weird recursion in the {AutoSchemaPlugin.__name__}.")
            return
        type(self)._ALREADY_DID = True

        search_path_entries = search_path.get_path()

        # WIP: Trying to infer the project root, configs dir from the Hydra context.
        # Currently hard-coded.

        for search_path_entry in search_path_entries:
            # TODO: There are probably lots of assumptions that are specific to the
            # ResearchTemplate repo that would need to be removed / generalized before we can make
            # this a pip-installable hydra plugin..
            if search_path_entry.provider != "main":
                continue

            if search_path_entry.path.startswith("pkg://"):
                configs_pkg = search_path_entry.path.removeprefix("pkg://")
                project_package = configs_pkg.split(".")[0]
                _project_module = importlib.import_module(project_package)
                assert (
                    _project_module.__file__
                    and Path(_project_module.__file__).name == "__init__.py"
                )
                repo_root = Path(_project_module.__file__).parent.parent
                configs_dir = repo_root / configs_pkg.replace(".", "/")
                if not (repo_root.is_dir() and configs_dir.is_dir()):
                    logger.warning(
                        f"Unable to add schemas to Hydra configs: "
                        f"Expected to find the project root directory at {repo_root} "
                        f"and the configs directory at {configs_dir}!"
                    )
                    continue
                # print(f"{self.fn=}, {_project_module}, {repo_root=}, {configs_dir=}")
                self.fn(
                    config_files=None,
                    repo_root=repo_root,
                    configs_dir=configs_dir,
                )


class AutoSchemaHandler(PatternMatchingEventHandler):
    def __init__(
        self,
        repo_root: Path,
        configs_dir: Path,
        schemas_dir: Path,
        regen_schemas: bool,
        stop_on_error: bool,
        quiet: bool,
        add_headers: bool | None,
    ):
        self.configs_dir = configs_dir
        super().__init__(
            patterns=[
                f"{self.configs_dir}/**.yaml",
                f"{self.configs_dir}/**.yml",
                f"{self.configs_dir}/**/*.yaml",
                f"{self.configs_dir}/**/*.yml",
            ],
            ignore_patterns=None,
            ignore_directories=True,
            case_sensitive=True,
        )
        self.repo_root = repo_root
        self.schemas_dir = schemas_dir
        self.regen_schemas = regen_schemas
        self.stop_on_error = stop_on_error
        self.quiet = quiet
        self.add_headers = add_headers

        self._last_updated: dict[Path, datetime.datetime] = {}

        # On startup, we could make a schema for every config file, right?
        add_schemas_to_all_hydra_configs(
            repo_root=repo_root,
            configs_dir=configs_dir,
            schemas_dir=schemas_dir,
            regen_schemas=regen_schemas,
            stop_on_error=stop_on_error,
            quiet=quiet,
            add_headers=add_headers,
        )

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        logger.debug(f"on_created event: {event.src_path}")
        if config_file := self._filter_config_file(event.src_path):
            logger.info(f"Config file was created: {config_file}")
            self._run(config_file)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        logger.debug(f"on_deleted event: {event.src_path}")

        # TODO: Remove the schema from the vscode settings.
        if config_file := self._filter_config_file(event.src_path):
            logger.info(f"Config file was deleted: {config_file}")
            self._remove_schema_file(config_file)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        logger.debug(f"on_modified event: {event.src_path}")

        if config_file := self._filter_config_file(event.src_path):
            if self._debounce(config_file):
                return
            logger.info(f"Config file was modified: {config_file}")
            self._run(config_file)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        logger.debug(f"on_moved event: {event.src_path}")

        source_file = self._filter_config_file(event.src_path)
        dest_file = self._filter_config_file(event.dest_path)
        if source_file and dest_file:
            logger.info(f"Config file was moved from {source_file} --> {dest_file}")
            self._remove_schema_file(source_file)
            self._run(dest_file)

        if dest_file:
            self._run(dest_file)

    def _filter_config_file(self, p: str | bytes) -> Path | None:
        config_file = Path(str(p))
        if config_file.suffix not in [".yaml", ".yml"]:
            return None
        if not config_file.is_relative_to(self.configs_dir):
            return None
        return config_file

    def _debounce(self, config_file: Path) -> bool:
        """Debouncing to prevent a loop caused by us modifying the config file to add a header(?).

        TODO: Also seems to be necessary when `self.add_header` is False! Why?

        Returns:
            `True` if the file was modified recently (by us), `False` otherwise.
        """
        if config_file not in self._last_updated:
            self._last_updated[config_file] = datetime.datetime.now()
            return False

        if (datetime.datetime.now() - self._last_updated[config_file]) < datetime.timedelta(
            seconds=5
        ):
            logger.debug(
                f"Config file {config_file} was modified very recently; not regenerating the schema."
            )
            return True
        self._last_updated[config_file] = datetime.datetime.now()
        return False

    def _run(self, config_file: Path) -> None:
        pretty_config_file_name = config_file.relative_to(self.configs_dir)
        schema_file = _get_schema_file_path(config_file, self.schemas_dir)

        logger.debug(f"Creating a schema for {pretty_config_file_name}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            config = _load_config(
                config_file, configs_dir=self.configs_dir, repo_root=self.repo_root
            )
        schema = create_schema_for_config(
            config, config_file=config_file, configs_dir=self.configs_dir, repo_root=self.repo_root
        )
        schema_file.parent.mkdir(exist_ok=True, parents=True)
        schema_file.write_text(json.dumps(schema, indent=2).rstrip() + "\n\n")
        # _set_is_incomplete_schema(schema_file, False)

        if not self.add_headers:
            try:
                _install_yaml_vscode_extension()
            except OSError:
                pass

            try:
                _add_schemas_to_vscode_settings(
                    {config_file: schema_file}, repo_root=self.repo_root
                )
            except Exception as exc:
                logger.error(
                    f"Unable to write schemas in the vscode settings file. "
                    f"Falling back to adding a header to config files. (exc={exc})"
                )
                if self.add_headers is not None:
                    # Unable to do it. Don't try to add headers, just return.
                    return
            else:
                # Success. Return.
                return

        logger.debug("Adding a header to the config file to point to the schema to use.")
        add_schema_header(config_file, schema_path=schema_file)

    def _remove_schema_file(self, config_file: Path) -> None:
        schema_file = _get_schema_file_path(config_file, self.schemas_dir)
        if self.add_headers:
            # Could also remove the schema file for this config file.
            logger.debug(
                f"Removing schema file associated with config {config_file}: {schema_file}"
            )
            schema_file.unlink()
        else:
            vscode_settings_file = self.repo_root / ".vscode" / "settings.json"
            if not vscode_settings_file.exists():
                # Weird.
                return
            vscode_settings = _read_json(vscode_settings_file)
            yaml_schemas = vscode_settings.get("yaml.schemas", {})
            assert isinstance(yaml_schemas, dict)
            if not yaml_schemas:
                # Weird also!
                return
            if (schema_key := str(schema_file.relative_to(self.repo_root))) not in yaml_schemas:
                # Weird also!
                return
            config_file_value = str(config_file)
            configs_with_schema: str | list[str] = yaml_schemas[schema_key]
            if isinstance(configs_with_schema, str):
                assert configs_with_schema == config_file_value
                # Weird!
                # Remove the schema from the dict.
                yaml_schemas.pop(schema_key)
            else:
                yaml_schemas[schema_key].remove(config_file_value)
            # NOTE: This doesn't work with comments.
            vscode_settings_file.write_text(json.dumps(vscode_settings, indent=4))


if __name__ == "__main__":
    main()

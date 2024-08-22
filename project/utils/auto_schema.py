"""Scripts that creates Schemas for hydra config files.

This is very helpful when using Hydra! It shows the user what options are available, along with their
description and default values, and displays errors if you have config files with invalid values.

## todos
- [ ] skip re-creating an existing schema unless a command-line flag is passed.
- [ ] Add schemas for all the nested dict entries in a config file if they have a _target_ (currently just the first level).
- [ ] Modify the schema to support omegaconf directives like ${oc.env:VAR_NAME} and our custom directives like ${instance_attr} and so on.
- [ ] Refine the schema for the `defaults` list to match what Hydra allows mnore closely.
- [ ] todo: Make a hydra plugin that creates the schemas for configs
"""

import argparse
import copy
import dataclasses
import inspect
import itertools
import json
import logging
import os.path
import shutil
import subprocess
import warnings
from collections.abc import Callable, MutableMapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

import docstring_parser as dp
import flax
import flax.linen
import flax.struct
import hydra.errors
import hydra.utils
import hydra_zen
import lightning.pytorch.callbacks
import pydantic
import pydantic.schema
import rich.logging
import tqdm
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from tqdm.rich import tqdm_rich
from typing_extensions import NotRequired, Required

from project.main import PROJECT_NAME
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.typing_utils import NestedMapping

logger = get_logger(__name__)

CONFIGS_DIR = REPO_ROOTDIR / PROJECT_NAME / "configs"


class PropertySchema(TypedDict, total=False):
    title: str
    type: Literal["string", "boolean", "object", "array", "integer"]
    description: str
    default: Any
    examples: list[str]
    deprecated: bool
    readOnly: bool
    writeOnly: bool
    const: Any
    enum: list[Any]


class ArrayPropertySchema(PropertySchema, total=False):
    type: Literal["array"]
    items: Required[PropertySchema]
    minItems: int
    maxItems: int
    uniqueItems: bool


class StringPropertySchema(PropertySchema, total=False):
    type: Literal["string"]
    pattern: str


class Schema(TypedDict, total=False):
    # "$defs":
    title: str
    description: str
    type: str
    properties: Required[MutableMapping[str, PropertySchema | ArrayPropertySchema]]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool]

    dependentRequired: NotRequired[MutableMapping[str, list[str]]]
    """ https://json-schema.org/understanding-json-schema/reference/conditionals#dependentRequired """


HYDRA_CONFIG_SCHEMA = Schema(
    title="Default Schema for any Hydra config file.",
    description=f"Schema created by the `{__file__}` script.",
    properties={
        "defaults": ArrayPropertySchema(
            title="Hydra defaults",
            description="Hydra defaults for this config. See https://hydra.cc/docs/advanced/defaults_list/",
            type="array",
            items=PropertySchema(type="string"),
            uniqueItems=True,
        ),
        "_target_": PropertySchema(
            type="string",
            title="Target",
            description="Target to instantiate.\nSee https://hydra.cc/docs/advanced/instantiate_objects/overview/",
        ),
        "_convert_": PropertySchema(
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


def main():
    logging.basicConfig(
        level=logging.INFO,
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
    # _root_logger = logging.getLogger("project")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPO_ROOTDIR)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    parser.add_argument("--schemas-dir", type=Path, default=None, required=False)
    parser.add_argument("--regen-schemas", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-q", "--quiet", dest="quiet", action=argparse.BooleanOptionalAction
    )
    verbosity_group.add_argument("-v", "--verbose", dest="verbose", action="count", default=0)

    args = parser.parse_args()

    repo_root = args.project_root
    configs_dir = args.configs_dir
    schemas_dir = args.schemas_dir
    regen_schemas = args.regen_schemas
    stop_on_error = args.stop_on_error

    if args.quiet:
        logger.setLevel(logging.NOTSET)
    elif args.verbose:
        if args.verbose >= 3:
            logger.setLevel(logging.DEBUG)
        elif args.verbose == 2:
            logger.setLevel(logging.INFO)
        else:
            assert args.verbose == 1
            logger.setLevel(logging.WARNING)

    run(
        repo_root=repo_root,
        configs_dir=configs_dir,
        schemas_dir=schemas_dir,
        regen_schemas=regen_schemas,
        stop_on_error=stop_on_error,
    )
    logger.info("Done updating the schemas for the Hydra config files.")


def run(
    repo_root: Path = REPO_ROOTDIR,
    configs_dir: Path = CONFIGS_DIR,
    schemas_dir: Path | None = None,
    regen_schemas: bool = False,
    stop_on_error: bool = False,
):
    if schemas_dir is None:
        schemas_dir = repo_root / ".schemas"

    if schemas_dir.is_relative_to(repo_root):
        _add_schemas_dir_to_gitignore(schemas_dir, repo_root=repo_root)

    config_files = list(configs_dir.rglob("*.yaml")) + list(configs_dir.rglob("*.yml"))

    if (_top_level_config := (configs_dir / "config.yaml")) in config_files:
        # todo: also add support for the top-level configs that are backed (or not) by a structured
        # config node.
        pass
        # config_files.remove(_top_level_config)

    if not config_files:
        warnings.warn(RuntimeWarning(f"Unable to find any config files {configs_dir}!"))
        return

    config_file_to_schema_file: dict[Path, Path] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
        pbar = tqdm_rich(config_files, desc="Creating schemas", total=len(config_files))

    for config_file in pbar:
        pretty_config_file_name = config_file.relative_to(configs_dir)
        schema_file = _get_schema_file_path(config_file, schemas_dir)

        if schema_file.exists() and not regen_schemas:
            if _is_incomplete_schema(schema_file):
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
                config = _load_config(config_file, configs_dir=configs_dir)
            schema = create_schema_for_config(
                config, config_file=config_file, configs_dir=configs_dir
            )
            schema_file.parent.mkdir(exist_ok=True, parents=True)
            schema_file.write_text(json.dumps(schema, indent=2) + "\n")
        except (
            pydantic.errors.PydanticSchemaGenerationError,
            hydra.errors.MissingConfigException,
        ) as exc:
            logger.warning(
                f"Unable to create a schema for config {pretty_config_file_name}: {exc}"
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

    # We will use option 1 if a `code` executable is found.
    set_schemas_in_vscode_settings_file = bool(shutil.which("code"))
    if set_schemas_in_vscode_settings_file:
        try:
            logger.debug(
                "Found the `code` executable, will add schema paths to the vscode settings."
            )
            _install_yaml_vscode_extension()
            _add_schemas_to_vscode_settings(config_file_to_schema_file, repo_root=repo_root)
            return
        except Exception as exc:
            logger.error(
                f"Unable to write schemas in the vscode settings file. "
                f"Falling back to adding a header to config files. (exc={exc})"
            )

    logger.debug("A headers to config files to point to the schemas to use.")
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


def _add_schemas_to_vscode_settings(
    config_file_to_schema_file: dict[Path, Path],
    repo_root: Path,
) -> None:
    # Make the vscode settings file if necessary:
    vscode_dir = repo_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True, parents=False)
    vscode_settings_file = vscode_dir / "settings.json"
    vscode_settings_file.touch(exist_ok=True)

    # TODO: What to do with comments? Ideally we'd keep them, right?
    logger.debug(f"Reading the VsCode settings file at {vscode_settings_file}.")
    vscode_settings_content = vscode_settings_file.read_text()
    # Remove any trailing commas from the content:
    vscode_settings_content = (
        vscode_settings_content.strip().removesuffix("}").rstrip().rstrip(",") + "}"
    )
    vscode_settings: dict[str, Any] = json.loads(vscode_settings_content)

    # Avoid the popup and do users a favour by disabling telemetry.
    vscode_settings.setdefault("redhat.telemetry.enabled", False)

    yaml_schemas_setting: dict[str, str | list[str]] = vscode_settings.setdefault(
        "yaml.schemas", {}
    )

    # Write all the schemas
    for config_file, schema_file in config_file_to_schema_file.items():
        assert schema_file.exists()

        schema_key = str(schema_file.relative_to(repo_root))
        # TODO: Make sure that this makes sense. What if people
        # open a folder like their $HOME in vscode, but the project is a subfolder?
        path_to_add = str(config_file.absolute())
        # path_to_add = str(config_file.relative_to(repo_root))

        if schema_key not in yaml_schemas_setting:
            yaml_schemas_setting[schema_key] = path_to_add
        elif isinstance(files_associated_with_schema := yaml_schemas_setting[schema_key], str):
            yaml_schemas_setting[schema_key] = sorted(
                set([files_associated_with_schema, path_to_add])
            )
        else:
            yaml_schemas_setting[schema_key] = sorted(
                set(files_associated_with_schema + [path_to_add])
            )

    vscode_settings_file.write_text(json.dumps(vscode_settings, indent=2))
    logger.info(f"Updated the yaml schemas in the vscode settings file at {vscode_settings_file}.")


def _get_schema_file_path(config_file: Path, schemas_dir: Path):
    config_group = config_file.parent
    schema_file = schemas_dir / f"{config_group.name}_{config_file.stem}_schema.json"
    return schema_file


def _get_shemas_for_hydra_configs(configs_dir: Path, regen_schemas: bool) -> dict[Path, Schema]:
    config_files_to_shemas: dict[Path, Schema] = {}

    for config_file in itertools.chain(configs_dir.rglob("*.yaml"), configs_dir.rglob("*.yml")):
        config_group = config_file.parent if config_file.parent != configs_dir else None
        if config_group is None and config_file.name == "config.yaml":
            # TODO: Special config file, gets validated against the structured config, not based on a target.
            continue

        try:
            logger.info(f"Creating a schema for {config_file.relative_to(configs_dir)}")
            config = _load_config(config_file, configs_dir=configs_dir)
            config_files_to_shemas[config_file] = create_schema_for_config(
                config, config_file=config_file, configs_dir=configs_dir
            )
        except Exception as e:
            logger.debug(
                f"Unable to update the schema for yaml config file {config_file.relative_to(configs_dir)}: {e}"
            )
            # raise
        else:
            logger.info(f"Updated schema for ./{_relative_to_cwd(config_file)}.")
    return config_files_to_shemas


def create_schema_for_config(
    config: dict | DictConfig, config_file: Path | None, configs_dir: Path | None
) -> Schema:
    """IDEA: Create a schema for the given config.

    - If you encounter a key, add it to the schema.
    - If you encounter a value with a _target_, use a dedicated function to get the schema for that target, and merge it into the current schema.
    - Only the top-level config (`config`) can have a `defaults: list[str]` key.
        - Should ideally load the defaults and merge this schema on top of them.
    """
    schema = copy.deepcopy(HYDRA_CONFIG_SCHEMA)
    schema["title"] = "Auto-generated schema."

    if config_file and config_file.exists() and configs_dir:
        # Config is an actual yaml file, not a structured config entry.
        p = config_file.relative_to(configs_dir) if configs_dir else config_file
        schema["title"] = f"Auto-generated schema for {p}"
        # note: the `defaults` list gets consumed by Hydra in `_load_config`, so we actually re-read the
        # config file to get the `defaults`, if present.
        _config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
        assert isinstance(_config, dict)
        # defaults = _config.get("defaults")
        assert "defaults" not in config
        assert config == _load_config(config_file, configs_dir=configs_dir)

        # if defaults:
        #     schema = _update_schema_from_defaults(
        #         config_file, schema=schema, defaults=defaults, configs_dir=configs_dir
        #     )

    if target_name := config.get("_target_"):
        # There's a '_target_' key at the top level in the config file.
        target = hydra.utils.get_object(target_name)
        schema["title"] = f"Auto-generated schema for {target}"
        schema["description"] = "Based on the target signature."
        schema["properties"]["_target_"] = PropertySchema(
            type="string",
            title="Target",
            const=target_name,
            # pattern=r"", # todo: Use a pattern to match python module import strings.
            description=(
                f"Target to instantiate, in this case: `{target_name}`\n"
                # f"* Source: <file://{relative_to_cwd(inspect.getfile(target))}>\n"
                # f"* Config file: <file://{config_file}>\n"
                f"See the Hydra docs for '_target_': https://hydra.cc/docs/advanced/instantiate_objects/overview/\n"
            ),
        )

        schema_from_target_signature = _get_schema_from_target(config)
        # logger.debug(f"Schema from signature of {target}: {schema_from_target_signature}")

        schema = _merge_dicts(
            schema_from_target_signature,
            schema,
            conflict_handler=overwrite,
        )

        return schema

    # Config file that contains entries that may or may not have a _target_.
    schema["additionalProperties"] = True

    for key, value in config.items():
        # Go over all the values in the config. If any of them have a `_target_`, then we can
        # add a schema at that entry.
        # TODO: make this work recursively?
        if isinstance(value, dict | DictConfig) and "_target_" in value.keys():
            target = hydra_zen.get_target(value)  # type: ignore
            schema_from_target_signature = _get_schema_from_target(value)
            logger.debug(
                f"Getting schema from target {value['_target_']} at key {key} in file {config_file}."
            )

            assert "properties" in schema_from_target_signature
            if key not in schema["properties"]:
                schema["properties"][key] = schema_from_target_signature
            else:
                raise NotImplementedError("todo: use merge_dicts here")

    return schema


def _update_schema_from_defaults(
    config_file: Path, schema: Schema, defaults: list[str | dict[str, str]], configs_dir: Path
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

        # try:
        default_config = _load_config(other_config_path, configs_dir=configs_dir)
        # except omegaconf.errors.MissingMandatoryValue:
        #     default_config = OmegaConf.load(other_config_path)

        schema_of_default = create_schema_for_config(
            config=default_config, config_file=other_config_path, configs_dir=configs_dir
        )

        logger.debug(f"Schema from default {default}: {schema_of_default}")
        logger.debug(f"Properties of {default=}: {list(schema_of_default['properties'].keys())}")

        schema = _merge_dicts(
            schema_of_default,
            schema,
            conflict_handlers={"title": overwrite, "description": overwrite},
        )
        # todo: deal with this one here.
        if schema.get("additionalProperties") is False:
            schema.pop("additionalProperties")
    return schema


def overwrite(val_a: Any, val_b: Any) -> Any:
    return val_b


def keep_previous(val_a: Any, val_b: Any) -> Any:
    return val_a


conflict_handlers: dict[str, Callable[[Any, Any], Any]] = {
    "_target_": overwrite,  # use the new target.
    "default": overwrite,  # use the new default?
}

D1 = TypeVar("D1", bound=NestedMapping)
D2 = TypeVar("D2", bound=NestedMapping)


def _merge_dicts(
    a: D1,
    b: D2,
    path: list[str] = [],
    conflict_handlers: dict[str, Callable[[Any, Any], Any]] = conflict_handlers,
    conflict_handler: Callable[[Any, Any], Any] | None = None,
) -> D1 | D2:
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
                    out[key] = specific_conflict_handler(a[key], b[key])
                elif conflict_handler:
                    out[key] = conflict_handler(a[key], b[key])

                # if any(key.split(".")[-1] == handler_name for  for prefix in ["_", "description", "title"]):
                #     out[key] = b[key]
                else:
                    raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            out[key] = copy.deepcopy(b[key])
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


def _load_config(config_path: Path, configs_dir: Path) -> DictConfig:
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra._internal.utils import create_automatic_config_search_path

    from project.main import PROJECT_NAME

    *config_groups, config_name = config_path.relative_to(configs_dir).with_suffix("").parts
    logger.debug(
        f"config_path: ./{_relative_to_cwd(config_path)}, {config_groups=}, {config_name=}, configs_dir: {configs_dir}"
    )
    config_group = "/".join(config_groups)

    if _has_package_global_line(config_path):
        # Tricky: Here we load the global config but with the given config as an override.
        search_path = create_automatic_config_search_path(
            calling_file=None,
            calling_module=None,
            # TODO: This doesn't seem to be working in unit tests?
            config_path=f"pkg://{PROJECT_NAME}.configs",
        )
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

    search_path = create_automatic_config_search_path(
        calling_file=None, calling_module=None, config_path=f"pkg://{PROJECT_NAME}.configs"
    )
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    top_config = config_loader.load_configuration(
        f"{config_group}/{config_name}", overrides=[], run_mode=RunMode.MULTIRUN
    )
    # Retrieve the sub-entry in the config and return it.
    config = top_config
    for config_group in config_groups:
        config = config[config_group]
    return config


def add_schema_header(config_file: Path, schema_path: Path) -> None:
    # TODO: THis line should be added **after** any comments like @package: global
    lines = config_file.read_text().splitlines(keepends=True)

    if config_file.parent is schema_path.parent:
        relative_path_to_schema_2 = "./" + schema_path.name
        # TODO: Unsure when this branch would be used, and if it would differ.
        assert False, (
            os.path.relpath(schema_path, start=config_file.parent),
            relative_path_to_schema_2,
        )
    else:
        relative_path_to_schema = os.path.relpath(schema_path, start=config_file.parent)

    # Remove any existing schema lines.
    lines = [
        line for line in lines if not line.strip().startswith("# yaml-language-server: $schema=")
    ]

    # NOTE: This line can be placed anywhere in the file, not necessarily needs to be at the top,
    # and the yaml vscode extension will pick it up.
    new_line = f"# yaml-language-server: $schema={relative_path_to_schema}\n"

    package_global_line: int | None = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # BUG: IF the schema line comes before a @package: global comment, then the @package: _global_
        # comment is ignored by Hydra.
        # Locate the last package line (a bit unnecessary, since there should only be one).
        if line.startswith("#") and line.removeprefix("#").strip().startswith("@package:"):
            package_global_line = i

    if package_global_line is None:
        # There's no package directive in the file.
        new_lines = [new_line, *lines]
    else:
        new_lines = lines.copy()
        # Insert the schema line after the package directive line.
        new_lines.insert(package_global_line + 1, new_line)

    result = "\n".join(new_lines) + "\n"
    if config_file.read_text() != result:
        config_file.write_text(result)


def _get_schema_from_target(config: dict | DictConfig) -> Schema:
    assert isinstance(config, dict | DictConfig)
    logger.debug(f"Config: {config}")
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
        from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme  # noqa
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
            zen_dataclass={"cls_name": target.__qualname__},
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
        assert "properties" in json_schema
    except pydantic.PydanticSchemaGenerationError as e:
        raise NotImplementedError(f"Unable to get the schema with pydantic: {e}")

    assert "properties" in json_schema

    docs_to_search: list[dp.Docstring] = []

    if inspect.isclass(target):
        for target_or_base_class in inspect.getmro(target):
            if class_docstring := inspect.getdoc(target_or_base_class):
                docs_to_search.append(dp.parse(class_docstring))
            if init_docstring := inspect.getdoc(target_or_base_class.__init__):
                docs_to_search.append(dp.parse(init_docstring))
    else:
        assert inspect.isfunction(target)
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

    def enum_schema(self, schema: "core_schema.EnumSchema") -> JsonSchemaValue:
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
            return super().enum_schema(slightly_changed_schema)
        return super().enum_schema(schema)


K = TypeVar("K")
V = TypeVar("V")


if __name__ == "__main__":
    main()

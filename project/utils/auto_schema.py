"""Scripts that creates Schemas for hydra config files.

This is very helpful when using Hydra! It shows the user what options are available, along with their
description and default values, and displays errors if you have config files with invalid values.

## todos
- [ ] skip re-creating an existing schema unless a command-line flag is passed.
- [ ] Add schemas for all the nested dict entries in a config file if they have a _target_ (currently just the first level).
- [ ] Modify the schema to support omegaconf directives like ${oc.env:VAR_NAME} and our custom directives like ${instance_attr} and so on.
- [ ] Refine the schema for the `defaults` list to match what Hydra allows mnore closely.
"""
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
import hydra.utils
import hydra_zen
import pydantic
import pydantic.schema
import rich.logging
from hydra.core.object_type import ObjectType
from hydra.core.plugins import Plugins
from hydra.plugins.config_source import ConfigResult, ConfigSource
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import NotRequired, Required

from project.main import PROJECT_NAME
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_config_utils import get_config_loader
from project.utils.typing_utils import NestedMapping, is_sequence_of

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



HYDRA_CONFIG_SCHEMA =  Schema(
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
    }
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
    _root_logger = logging.getLogger("project")

    repo_root: Path = REPO_ROOTDIR
    configs_dir: Path = CONFIGS_DIR
    schemas_dir: Path = REPO_ROOTDIR / ".schemas"

    config_files_to_schemas = _get_shemas_for_hydra_configs(configs_dir=configs_dir)

    if not config_files_to_schemas:
        logger.warning(f"Unable to deduce a schema to use for any configs in {configs_dir}.")
        return

    schemas_dir.mkdir(exist_ok=True)
    if schemas_dir.is_relative_to(REPO_ROOTDIR):
        _rel = schemas_dir.relative_to(REPO_ROOTDIR)
        _gitignore_file = REPO_ROOTDIR/".gitignore"
        if not any(line.startswith(str(_rel)) for line in _gitignore_file.read_text().splitlines()):
            logger.info(f"Adding entry in .gitignore for the schemas directory ({schemas_dir})")
            with _gitignore_file.open("a") as f:
                f.write(f"{_rel}\n")

    config_file_to_schema_and_schema_file: dict[Path, tuple[Path, Schema]] = {}

    # Write all the schemas to the schema directory.
    for config_file, schema in config_files_to_schemas.items():
        schema_file = _get_schema_file_path(config_file, schema, schemas_dir)

        _content = json.dumps(schema, indent=2)
        if not schema_file.exists():
            logger.info(f"Writing a new schema for {relative_to_cwd(config_file)} at {relative_to_cwd(schema_file)}.")
            schema_file.write_text(_content + "\n")
        elif schema_file.read_text().strip() != _content:
            logger.info(f"Updating the schema for {relative_to_cwd(config_file)} at {relative_to_cwd(schema_file)}.")
            schema_file.write_text(_content + "\n")
        else:
            logger.debug(f"Schema at {relative_to_cwd(schema_file)} is already up-to-date.")

        config_file_to_schema_and_schema_file[config_file] = (schema_file, schema)

    # Option 1: Add a vscode setting that associates the schema file with the yaml files. (less intrusive perhaps).
    # Option 2: Add a header to the yaml files that points to the schema file.

    # We will use option 1 if a `code` executable is found.
    set_schemas_in_vscode_settings_file = bool(shutil.which("code"))
    if set_schemas_in_vscode_settings_file:
        logger.debug("Found the `code` executable, will add schema paths to the vscode settings.")
        _install_yaml_vscode_extension()
        _add_schemas_to_vscode_settings(config_files_to_schemas, repo_root=repo_root, schemas_dir=schemas_dir)
    else:
        logger.debug(
            "Did not find the `code` executable on this machine. Will add headers to config files to point to the schemas to use."
        )
        for config_file, (schema_file, schema) in config_file_to_schema_and_schema_file.items():
            _add_schema_header(config_file, schema_path=schema_file)

    logger.info("Done updating the schemas for the Hydra config files.")

def relative_to_cwd(p: str | Path):
    return Path(p).relative_to(Path.cwd())

def _install_yaml_vscode_extension():
    logger.debug("Running `code --install-extension redhat.vscode-yaml` to install the yaml extension for vscode.")
    output = subprocess.check_output(("code", "--install-extension", "redhat.vscode-yaml"), text=True)
    logger.debug(output)

def _add_schemas_to_vscode_settings(config_files_to_schemas: dict[Path, Schema], repo_root: Path, schemas_dir: Path) -> None:
    # Make the vscode settings file if necessary:
    vscode_dir = repo_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True, parents=False)
    vscode_settings_file = vscode_dir / "settings.json"
    vscode_settings_file.touch(exist_ok=True)

    # TODO: What to do with comments? Ideally we'd keep them, right?
    logger.info(f"Reading the VsCode settings file at {vscode_settings_file}.")
    vscode_settings_content = vscode_settings_file.read_text()
    # Remove any trailing commas from the content:
    vscode_settings_content = vscode_settings_content.strip().removesuffix("}").rstrip().rstrip(",") + "}"
    vscode_settings: dict[str, Any] = json.loads(vscode_settings_content)

    # Avoid the popup and do users a favour by disabling telemetry.
    vscode_settings.setdefault("redhat.telemetry.enabled", False)

    yaml_schemas_setting: dict[str, str | list[str]] = vscode_settings.setdefault(
        "yaml.schemas", {}
    )

    # Write all the schemas
    for config_file, schema in config_files_to_schemas.items():
        schema_file = _get_schema_file_path(config_file, schema, schemas_dir)
        schema_file.write_text(json.dumps(schema, indent=2) + "\n")

        schema_key = str(schema_file.relative_to(repo_root))
        # TODO: Make sure that this makes sense. What if people
        # open a folder like their $HOME in vscode, but the project is a subfolder?
        path_to_add = str(config_file.absolute())
        # path_to_add = str(config_file.relative_to(repo_root))

        if schema_key not in yaml_schemas_setting:
            yaml_schemas_setting[schema_key] = path_to_add
        elif isinstance(files_associated_with_schema := yaml_schemas_setting[schema_key], str):
            yaml_schemas_setting[schema_key] = sorted(set([files_associated_with_schema, path_to_add]))
        else:
            yaml_schemas_setting[schema_key] = sorted(set(files_associated_with_schema + [path_to_add]))

    vscode_settings_file.write_text(json.dumps(vscode_settings, indent=2))
    logger.info(f"Updated the yaml schemas in the vscode settings file at {vscode_settings_file}.")


def _get_schema_file_path(config_file: Path, schema: Schema, schemas_dir: Path):
    config_group = config_file.parent
    schema_file = schemas_dir / f"{config_group.name}_{config_file.stem}_schema.json"
    return schema_file


def _get_shemas_for_hydra_configs(configs_dir: Path) -> dict[Path, Schema]:
    config_files_to_shemas: dict[Path, Schema] = {}

    for config_file in itertools.chain(configs_dir.rglob("*.yaml"), configs_dir.rglob("*.yml")):
        config_group = config_file.parent if config_file.parent != configs_dir else None
        if config_group is None and config_file.name == "config.yaml":
            # TODO: Special config file, gets validated against the structured config, not based on a target.
            continue

        try:
            logger.info(f"Creating a schema for {config_file.relative_to(configs_dir)}")
            config = _load_config(config_file, configs_dir=configs_dir)
            config_files_to_shemas[config_file] = _create_schema_for_config(config, config_file=config_file, configs_dir=configs_dir)
        except Exception as e:
            logger.debug(f"Unable to update the schema for yaml config file {config_file.relative_to(configs_dir)}: {e}")
            # raise
        else:
            logger.info(f"Updated schema for ./{relative_to_cwd(config_file)}.")
    return config_files_to_shemas


def _create_schema_for_config(config: dict | DictConfig, config_file: Path, configs_dir: Path) -> Schema:
    """IDEA: Create a schema for the given config.

    - If you encounter a key, add it to the schema.
    - If you encounter a value with a _target_, use a dedicated function to get the schema for that target, and merge it into the current schema.
    - Only the top-level config (`config`) can have a `defaults: list[str]` key.
        - Should ideally load the defaults and merge this schema on top of them.
    """
    schema = copy.deepcopy(HYDRA_CONFIG_SCHEMA)
    schema["title"] = f"Auto-generated schema for {config_file.relative_to(configs_dir)}"


    if config_file.exists():
        # Config is an actual yaml file, not a structured config entry.
        # note: the `defaults` list gets consumed by Hydra in `_load_config`, so we actually re-read the
        # config file to get the `defaults`, if present.
        _config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
        assert isinstance(_config, dict)
        defaults = _config.get("defaults")
        if defaults:
            schema = _update_schema_from_defaults(config_file, schema=schema, defaults=defaults, configs_dir=configs_dir)

    if target_name := config.get("_target_"):
        # There's a '_target_' key at the top level in the config file.
        target = hydra.utils.get_object(target_name)
        schema["title"] = f"Auto-generated schema for {target}"
        schema["description"] = f"Based on the signature of {target}."
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
        if config.get("_partial_"):
            # todo: add a special marker that allows extra fields?
            schema["required"] = []

        # if the target takes **kwargs, then we don't restrict additional properties.
        schema["additionalProperties"] = inspect.getfullargspec(target).varkw is not None

        value_schema = _get_schema_from_target(config, config_file=config_file)
        logger.debug(f"Schema from target {config['_target_']}: {value_schema}")

        schema = _merge_dicts(
            value_schema,
            schema,
            conflict_handler=overwrite,
        )
        schema = _adapt_schema_for_hydra(config_file, config, schema)

        return schema

    # Config file that contains entries that may or may not have a _target_.
    schema["additionalProperties"] = True
    for key, value in config.items():
        # Go over all the values in the config. If any of them have a `_target_`, then we can
        # add
        if isinstance(value, dict | DictConfig) and "_target_" in value.keys():
            # schema_from_target = get_schema_from_target(value, config_file=config_file)
            target = hydra_zen.get_target(value)  # type: ignore
            value_schema = _get_schema_from_target(value, config_file=config_file)
            logger.debug(
                f"Getting schema from target {value['_target_']} at key {key} in file {config_file}."
            )

            assert "properties" in value_schema
            if key not in schema["properties"]:
                schema["properties"][key] = value_schema
            else:
                raise NotImplementedError("todo: use merge_dicts here")

    return schema


def _update_schema_from_defaults(config_file: Path, schema: Schema, defaults: list[str], configs_dir: Path):
    defaults_list = defaults
    if not is_sequence_of(defaults_list, str):
        # TODO: If there's a default like `- callbacks: foobar.yml` in `trainer/default.yaml`, then
        # perhaps we could also add a schema for one of the entries in the config file if there is a default
        logger.error(f"The defaults list in {config_file} doesn't contain only strings! Skipping.")
        return schema
    for default in defaults_list:
        if default == "_self_":  # todo: does this actually make sense?
            continue
        assert not default.startswith("/")

        if default.endswith((".yaml", ".yml")):
            default = default.removesuffix(".yaml").removesuffix(".yml")

        other_config_path = config_file.parent / f"{default}.yaml"
        config = _load_config(other_config_path, configs_dir=configs_dir)
        # raise RuntimeError(
        #     f"Can't find the config file for default {default!r} in config {config_file}: "
        #     f"{other_config_path} doesn't exist!"
        # )
        schema_of_default = _create_schema_for_config(config=config, config_file=other_config_path, configs_dir=configs_dir)

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


def _load_config(config_path: Path, configs_dir: Path) -> DictConfig:
    *config_groups, config_name = config_path.relative_to(configs_dir).with_suffix("").parts
    logger.debug(f"{config_path=}, {config_groups=}, {config_name=}")
    config_group = "/".join(config_groups)
    top_config = get_config_loader().load_configuration(
        f"{config_group}/{config_name}", overrides=[], run_mode=RunMode.RUN
    )
    config = top_config
    for config_group in config_groups:
        config = config[config_group]
    return config


class _AutoSchemaPlugin(ConfigSource):
    # todo: Perhaps we can make a hydra plugin with the auto-schema stuff?
    def __init__(self, provider: str, path: str) -> None:
        super().__init__(provider=provider, path=path)
        logger.info(f"{provider=}, {path=}")

    @staticmethod
    def scheme() -> str:
        return "auto_schema"

    def load_config(self, config_path: str) -> ConfigResult:
        _name = self._normalize_file_name(config_path)
        raise NotImplementedError(config_path)

    def is_group(self, config_path: str) -> bool:
        raise NotImplementedError(config_path)

    def is_config(self, config_path: str) -> bool:
        raise NotImplementedError(config_path)

    def available(self) -> bool:
        """
        :return: True is this config source is pointing to a valid location
        """
        return True
        raise NotImplementedError()

    def list(self, config_path: str, results_filter: ObjectType | None) -> list[str]:
        """List items under the specified config path.

        :param config_path: config path to list items in, examples: "", "foo", "foo/bar"
        :param results_filter: None for all, GROUP for groups only and CONFIG for configs only
        :return: a list of config or group identifiers (sorted and unique)
        """
        raise NotImplementedError(config_path, results_filter)


def register_auto_schema_plugin() -> None:
    Plugins.instance().register(_AutoSchemaPlugin)


def _add_schema_header(config_file: Path, schema_path: Path) -> None:
    input_lines = config_file.read_text().splitlines(keepends=True)
    relative_path_to_schema = os.path.relpath(schema_path, start=config_file.parent)
    if config_file.parent is schema_path.parent:
        relative_path_to_schema = "./" + schema_path.name
    new_first_line = f"# yaml-language-server: $schema={relative_path_to_schema}\n"
    # todo; remove leading empty lines.
    if input_lines[0].startswith("# yaml-language-server: $schema="):
        output_lines = [new_first_line, *input_lines[1:]]
    else:
        output_lines = [new_first_line, *input_lines]

    with config_file.open("w") as f:
        f.writelines(output_lines)


def _get_schema_from_target(config: dict | DictConfig, config_file: Path) -> Schema:
    assert isinstance(config, dict | DictConfig)
    logger.debug(f"Config: {config}")
    target = hydra_zen.get_target(config)  # type: ignore

    if inspect.isclass(target) and issubclass(target, flax.linen.Module):
        object_type = hydra_zen.builds(
            target,
            populate_full_signature=True,
            hydra_recursive=False,
            hydra_convert="all",
            zen_exclude=["parent"],
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
                mode="serialization", schema_generator=_MyGenerateJsonSchema, by_alias=False,
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
            property_dict["description"] = f"The {property_name} parameter of the {target.__qualname__}."


    # Add field docstrings as descriptions in the schema!
    # json_schema = _update_schema_with_descriptions(object_type, json_schema=json_schema)

    return json_schema


# TODO: read this:
# https://suneeta-mall.github.io/2022/03/15/hydra-pydantic-config-management-for-training-application.html#hydra-directives
# https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw
def _adapt_schema_for_hydra(
    input_file: Path, config: dict | DictConfig, schema_from_pydantic: Schema
):
    """TODO: Adapt the schema to be better adapted for Hydra configs.

    TODOs:
    - [ ] defaults should always be accepted as a field.
    - [ ] _partial_ should make it so there are no mandatory fields
    - [ ] Unexpected extra fields should not be allowed
    """
    # TODO: This generated schema does not seem that well-adapted for Hydra, actually.
    schema = copy.deepcopy(schema_from_pydantic)

    if hydra_zen.is_partial_builds(config):
        # todo: add a special marker that allows extra fields?
        schema["required"] = []

    if _target_has_var_kwargs(config):
        # if the target takes **kwargs, then we don't restrict additional properties.
        schema["additionalProperties"] = True
    else:
        schema["additionalProperties"] = False
    return schema


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

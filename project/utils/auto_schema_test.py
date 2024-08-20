import json
from pathlib import Path

import hydra_zen
import pytest
from hydra.core.plugins import Plugins
from hydra.plugins.config_source import ConfigSource
from hydra.test_utils.config_source_common_tests import ConfigSourceTestSuite
from pytest_regressions.file_regression import FileRegressionFixture

from project.utils.auto_schema import (
    AutoSchemaPlugin,
    add_schema_header,
    create_schema_for_config,
    get_schema_from_target,
)
from project.utils.env_vars import REPO_ROOTDIR


@pytest.mark.xfail(raises=(NotImplementedError, ValueError), reason="Not implemented yet.")
@pytest.mark.parametrize(
    ("type_", "path"), [(AutoSchemaPlugin, "project://project.utils.auto_schema")]
)
class TestAutoSchemaPlugin(ConfigSourceTestSuite): ...


def test_discovery() -> None:
    # Test that this config source is discoverable when looking at config sources
    assert AutoSchemaPlugin.__name__ in [
        x.__name__ for x in Plugins.instance().discover(ConfigSource)
    ]


# @dataclass
class Foo:
    # some_integer: int
    # optional_str: str = "bob"
    def __init__(self, some_integer: int, optional_str: str = "bob"):
        self.some_integer = some_integer
        self.optional_str = optional_str


@pytest.mark.parametrize(
    "input_file",
    [file for file in (REPO_ROOTDIR / "project/configs").rglob("*.yaml")],
    ids=lambda x: str(x.relative_to(REPO_ROOTDIR / "project/configs")),
)
def test_create_schema_for_config(
    input_file: Path,
    tmp_path: Path,
    file_regression: FileRegressionFixture,
    original_datadir: Path,
):
    if input_file.name == "config.yaml":
        pytest.skip(
            reason="TODO: handle the top-level config later, seems harder to work on for a first POC"
        )
    config_file = original_datadir / input_file.name
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config_file.write_text(input_file.read_text())

    schema = create_schema_for_config(input_file)
    schema_path = (original_datadir / input_file.name).with_suffix(".json")
    add_schema_header(config_file, schema_path)

    schema = get_schema_from_target(config_file)
    file_regression.check(json.dumps(schema, indent=2), fullpath=schema_path, extension=".json")


@pytest.mark.skip(reason="Skipping for now in favour of the test above.")
@pytest.mark.parametrize(
    "input_file",
    [file for file in (REPO_ROOTDIR / "project/configs").rglob("*.yaml")],
    ids=lambda x: str(x.relative_to(REPO_ROOTDIR / "project/configs")),
)
def test_get_schema(
    input_file: Path,
    tmp_path: Path,
    file_regression: FileRegressionFixture,
    original_datadir: Path,
):
    # *config_groups, config_name = input_file.relative_to(CONFIGS_DIR).with_suffix("").parts
    # config_group = "/".join(config_groups)

    # config_name = str(input_file.relative_to(REPO_ROOTDIR / "project/configs"))
    try:
        # todo: this is dumb, the _target_ could be in the defaults list!
        _config = hydra_zen.load_from_yaml(input_file)
        _target = hydra_zen.get_target(_config)  # type: ignore
    except TypeError:
        pytest.skip(reason=f"Config at {input_file} doesn't have a target.")

    config_file = original_datadir / input_file.name
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config_file.write_text(input_file.read_text())

    schema_path = (original_datadir / input_file.name).with_suffix(".json")
    add_schema_header(config_file, schema_path)

    schema = get_schema_from_target(config_file)

    file_regression.check(json.dumps(schema, indent=2), fullpath=schema_path, extension=".json")

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from pytest_regressions.file_regression import FileRegressionFixture

from .auto_schema import add_schema_header, create_schema_for_config, main


class Foo:
    def __init__(self, bar: str):
        """Description of the `Foo` class.

        Args:
            bar: Description of the `bar` argument.
        """
        self.bar = bar


class Bar(Foo):
    """Docstring of the Bar class.

    Args:
        baz: description of the `baz` argument from the cls docstring instead of the init docstring.
    """

    def __init__(self, bar: str, baz: int):
        # no docstring here.
        super().__init__(bar=bar)
        self.baz = baz


_this_file = Path(__file__)
_config_dir = (_this_file.parent / _this_file.name).with_suffix("")
test_files = list(_config_dir.rglob("*.yaml"))


@pytest.mark.parametrize("config_file", test_files, ids=[f.name for f in test_files])
def test_make_schema(config_file: Path, file_regression: FileRegressionFixture):
    """Test that creates a schema for a config file and saves it next to it.

    (in the test folder).
    """
    schema_file = config_file.with_suffix(".json")

    config = yaml.load(config_file.read_text(), yaml.FullLoader)
    schema = create_schema_for_config(
        config=config, config_file=config_file, configs_dir=_config_dir
    )

    add_schema_header(config_file, schema_path=schema_file)
    file_regression.check(
        json.dumps(schema, indent=2) + "\n", fullpath=schema_file, extension=".json"
    )


def test_can_run_via_cli():
    """Actually run the command on the repo from the CLI."""
    # Run programmatically instead of with a subproc4ess so we can get nice coverage stats.
    main(["."])  # assuming we're at the project root directory.


def test_run_via_cli_without_errors():
    """Checks that the command completes without errors."""
    # Run programmatically instead of with a subproc4ess so we can get nice coverage stats.
    main([".", "--stop-on-error"])  # assuming we're at the project root directory.


@pytest.mark.xfail(
    reason="Rye isn't used anymore. TODO: Figure out the uv equivalent of rye scripts."
)
def test_run_via_rye_script():
    """Actually run the command on the repo, via the `[tool.rye.scripts]` entry in
    pyproject.toml."""
    # Run once so we can get nice coverage stats.
    subprocess.check_call(["rye", "run", "auto_schema"], text=True)


@pytest.mark.xfail(
    reason="Rye isn't used anymore. TODO: Figure out the uv equivalent of rye scripts."
)
def test_run_via_rye_script_without_errors():
    """Actually run the command on the repo, via the `[tool.rye.scripts]` entry in
    pyproject.toml."""
    # Run once so we can get nice coverage stats.
    subprocess.check_call(["rye", "run", "auto_schema", "--stop-on-error"], text=True)

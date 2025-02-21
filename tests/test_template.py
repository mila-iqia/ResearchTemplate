import subprocess
import sys
from logging import getLogger
from pathlib import Path

import pytest
import tomli
from copier import Worker

import project
import project.algorithms
from project.utils.testutils import IN_GITHUB_CLOUD_CI, IN_SELF_HOSTED_GITHUB_CI

logger = getLogger(__name__)

example_folder = Path(project.algorithms.__file__).parent
examples: list[str] = [
    p.relative_to(example_folder).stem
    for p in example_folder.glob("*.py")
    if not (p.name == "no_op.py" or p.name.endswith("_test.py") or p.name.startswith("_"))
]


@pytest.fixture(params=[None] + sorted(examples))
def examples_to_include(request: pytest.FixtureRequest):
    """Fixture that provides the examples that would be selected by the users."""
    example = request.param
    # For now, we run with either no examples, or a single example from the list. We don't test
    # combinations of examples.
    return [] if example is None else [example]


@pytest.mark.skipif(
    IN_GITHUB_CLOUD_CI,
    reason="TODO: lots of issues on GitHub CI (commit author, can't install other Python versions).",
)
@pytest.mark.skipif(sys.platform == "win32", reason="The template isn't supported on Windows.")
@pytest.mark.parametrize(
    "python_version",
    [
        # These can be very slow but are super important!
        # don't run these unless --slow argument is passed to pytest, to save some time.
        # TODO: This seems to be the only one that works in the CI atm, because:
        # - UV seems unable to download other python versions?
        # - Python 3.11 and 3.12 aren't able to install orion atm.
        "3.10",
        pytest.param("3.11", marks=pytest.mark.slow),
        pytest.param(
            "3.12",
            marks=[] if IN_SELF_HOSTED_GITHUB_CI else pytest.mark.slow,
        ),
        pytest.param(
            "3.13",
            marks=[
                pytest.mark.slow,
                pytest.mark.xfail(
                    reason="TODO: Update dependencies (torch, jax, t-j-i, mujoco, ...) for python 3.13",
                    strict=True,
                ),
            ],
        ),
    ],
)
def test_template(
    examples_to_include: list[str],
    python_version: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Run Copier programmatically to test the the setup for new projects.

    NOTE: This test is slow at first, as it might fill up your UV cache with torch / jax / etc for
    each python version!
    """
    tmp_project_dir = tmp_path / "new_project"

    # This is like doing `copier copy --trust gh:mila-iqia/ResearchTemplate new_project`
    # Except we use `--vcs-ref HEAD` and supply the answers (`-d project_name=new_project ...`)

    with Worker(
        src_path=".",
        dst_path=tmp_project_dir,
        vcs_ref="HEAD",
        defaults=True,
        data={
            "project_name": "new_project",
            "user_name": "John Doe",
            "examples_to_include": examples_to_include,
            "github_user": "johndoe",
            "python_version": python_version,
        },
        unsafe=True,
    ) as worker:
        worker.run_copy()
        # Note: here we just collect tests.
        command = ["uv", "run", "pytest", "-v", "--collect-only"]
        logger.info(f"Running: command `{' '.join(command)}`")
        result = subprocess.run(
            command,
            cwd=tmp_project_dir,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print("STDOUT: ", result.stdout)
            print("STDERR: ", result.stderr)
            pytest.fail(f"Failed to collect tests in the new project: {result.stdout}")


def test_templated_dependencies_are_same_as_in_project():
    """Test that the dependencies listed in the `pyproject.toml` of the template are the same as in
    the templated pyproject.toml."""
    project_toml = tomli.loads(Path("pyproject.toml").read_text())
    project_toml_template = tomli.loads(Path("pyproject.toml.jinja").read_text())
    assert sorted(project_toml["project"]["dependencies"]) == sorted(
        project_toml_template["project"]["dependencies"]
    )

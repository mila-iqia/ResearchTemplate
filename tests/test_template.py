import subprocess
import sys
from pathlib import Path

import pytest
from copier import Worker

import project
import project.algorithms

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


@pytest.mark.skipif(sys.platform == "win32", reason="The template isn't supported on Windows.")
@pytest.mark.parametrize(
    "python_version",
    [
        # don't run these unless --slow argument is passed to pytest, to save some time.
        pytest.param("3.10", marks=pytest.mark.slow),
        pytest.param("3.11", marks=pytest.mark.slow),
        "3.12",
        pytest.param(
            "3.13",
            marks=[
                pytest.mark.slow,
                pytest.mark.xfail(
                    raises=subprocess.CalledProcessError,
                    reason="TODO: Update dependencies (torch, jax, t-j-i, mujoco, ...) for python 3.13",
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
        print(f"Running: command {' '.join(command)}")
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

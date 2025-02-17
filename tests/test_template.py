import subprocess
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
    return [] if example is None else [example]


@pytest.mark.parametrize(
    "python_version",
    [
        # don't run these unless --slow argument is passed to pytest, to save some time.
        pytest.param("3.10", marks=pytest.mark.slow),
        pytest.param("3.11", marks=pytest.mark.slow),
        "3.12",
        pytest.param(
            "3.13",
            marks=pytest.mark.xfail(
                reason="TODO: Update dependencies (torch, jax, t-j-i, mujoco, ...) for python 3.13"
            ),
        ),
    ],
)
def test_template(examples_to_include: list[str], python_version: str, tmp_path: Path):
    """Run Copier programmatically to test the the setup for new projects."""
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
            "your_name": "John Doe",
            "examples_to_include": examples_to_include,
            "github_username": "johndoe",
            "python_version": python_version,
        },
        unsafe=True,
    ) as worker:
        worker.run_copy()
        # Note: here we just collect tests.
        out = subprocess.check_output(
            ["uv", "run", "pytest", "-v", "--collect-only"],
            cwd=tmp_project_dir,
            text=True,
            # capture_output=True,
            # stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(out)

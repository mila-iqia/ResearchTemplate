import contextlib
import dataclasses
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Protocol

import pytest
import tomli
import yaml
from copier import Worker
from copier.vcs import get_git
from plumbum import local

import project
import project.algorithms
from project.main import REPO_ROOTDIR
from project.utils.testutils import IN_GITHUB_CLOUD_CI

logger = getLogger(__name__)

example_folder = Path(project.algorithms.__file__).parent
examples: list[str] = [
    p.relative_to(example_folder).stem
    for p in example_folder.glob("*.py")
    if not (
        p.name == "no_op.py"
        or p.name.endswith("_test.py")
        or p.name.endswith("_tests.py")
        or p.name.startswith("_")
    )
]

_DEFAULT_PYTHON_VERSION = "3.10"
"""The default choice of python version in the copier.yaml file."""

_project_fixture_scope = "module"
"""The scope of the fixtures that are used to set up the project.

todo: ideally we'd only setup the project once per initial version and reuse it for all tests, but
some tests update the folder (e.g. `test_update_project`), so we can't reuse the fixture atm.
"""


@pytest.fixture(
    params=[[], *[[example] for example in examples], examples],
    scope=_project_fixture_scope,
    ids=["none", *examples, "all"],
)
def examples_to_include(request: pytest.FixtureRequest):
    """Fixture that provides the examples that would be selected by the users."""
    # By default, select all of the examples.
    return request.param


def test_templated_dependencies_are_same_as_in_project():
    """Test that the dependencies listed in the `pyproject.toml` of the template are the same as in
    the templated pyproject.toml."""
    project_toml = tomli.loads(Path("pyproject.toml").read_text())
    project_toml_template = tomli.loads(Path("pyproject.toml.jinja").read_text())
    assert sorted(project_toml["project"]["dependencies"]) == sorted(
        project_toml_template["project"]["dependencies"]
    )


all_template_versions: list[str] = subprocess.getoutput("git tag --sort=-creatordate").split("\n")
# todo: Further reduce the number of tests by only testing certain transitions (to the next major
# or minor version for example).
# starting_versions = [
#     tag
#     for tag in all_template_versions
#     if not (v := version.parse(tag)).is_prerelease or v.is_devrelease
# ]
# todo: Why would we keep testing updating to older versions?
from_to: dict[str, str] = {
    from_version: to_version
    for i, from_version in enumerate(all_template_versions)
    for to_version in (all_template_versions[i + 1 :] + ["HEAD"])
}


@pytest.fixture(scope=_project_fixture_scope)
def python_version(request: pytest.FixtureRequest):
    return getattr(request, "param", _DEFAULT_PYTHON_VERSION)


@pytest.fixture(ids=str, scope=_project_fixture_scope)
def template_version_used(request: pytest.FixtureRequest) -> str:
    """Return the version of the template to use when setting up the new project."""
    return getattr(request, "param", "HEAD")


@dataclasses.dataclass(frozen=True)
class CopierAnswers:
    """Answers to the questions that Copier asks the user when generating a new project."""

    project_name: str = "project"
    module_name: str = "project"
    user_name: str = "John Doe"
    examples_to_include: list[str] = dataclasses.field(default_factory=list)
    github_user: str = "johndoe"
    python_version: str = _DEFAULT_PYTHON_VERSION


@pytest.fixture(scope=_project_fixture_scope)
def copier_answers(
    python_version: str,
    examples_to_include: list[str],
):
    return CopierAnswers(
        project_name="NewProject",
        module_name="new_project",
        user_name="John Doe",
        examples_to_include=examples_to_include,
        github_user="johndoe",
        python_version=python_version,
    )


@pytest.fixture(scope="function")
def project_root(template_version_used: str, tmp_path_factory: pytest.TempPathFactory):
    tmp_project_dir = tmp_path_factory.mktemp(f"project_{template_version_used}_test")
    return tmp_project_dir


@pytest.fixture(scope="function")
def temporarily_set_git_config_for_commits(project_root: Path):
    # On GitHub Actions, we need to set user.name and user.email for the `git commit` commands
    # to succeed.
    if not IN_GITHUB_CLOUD_CI:
        yield
        return
    # note: doesn't actually change anything to cd to the project root since we modify the global
    # git config (which sucks, but the project root is not yet a git repo at this point, so we
    # can't set the local config).
    # git = get_git().with_cwd(project_root)
    # git_user_name_before = git("config", "--get", "user.name")
    # git_user_email_before = git("config", "--get", "user.email")
    git_user_name_before = subprocess.getoutput(
        ("git", "config", "--global", "--get", "user.name")
    )
    git_user_email_before = subprocess.getoutput(
        ("git", "config", "--global", "--get", "user.email")
    )
    try:
        # git("config", "user.name", "your-name")
        # git("config", "user.email", "your-email@email.com")
        subprocess.check_call(("git", "config", "--global", "user.name", "your-name"))
        subprocess.check_call(("git", "config", "--global", "user.email", "your-email@email.com"))
        yield
    finally:
        # git("config", "user.name", git_user_name_before)
        # git("config", "user.email", git_user_email_before)
        if git_user_email_before:
            subprocess.check_call(
                ("git", "config", "--global", "user.email", git_user_email_before)
            )
        else:
            subprocess.check_call(("git", "config", "--global", "--unset", "user.email"))
        if git_user_name_before:
            subprocess.check_call(("git", "config", "--global", "user.name", git_user_name_before))
        else:
            subprocess.check_call(("git", "config", "--global", "--unset", "user.name"))
    return


@pytest.fixture(scope="function")
def project_from_template(
    template_version_used: str,
    project_root: Path,
    copier_answers: CopierAnswers,
    temporarily_set_git_config_for_commits: None,
):
    """Fixture that provides the project at a given version."""
    logger.info(
        f"Setting up a project at {project_root} using the template at version "
        f"{template_version_used} with answers: {copier_answers}"
    )
    with Worker(
        src_path="." if template_version_used == "HEAD" else "gh:mila-iqia/ResearchTemplate",
        dst_path=project_root,
        vcs_ref=template_version_used,
        defaults=True,
        data=dataclasses.asdict(copier_answers),
        unsafe=True,
    ) as worker:
        worker.run_copy()
        assert worker.dst_path == project_root and project_root.exists()
        yield project_root


# @pytest.mark.skipif(
#     IN_GITHUB_CLOUD_CI,
#     reason="TODO: lots of issues on GitHub CI (commit author, can't install other Python versions).",
# )
@pytest.mark.skipif(sys.platform == "win32", reason="The template isn't supported on Windows.")
@pytest.mark.parametrize(
    examples_to_include.__name__,
    [[], *[[example] for example in examples], examples],
    indirect=True,
    ids=["none", *examples, "all"],
)
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
        pytest.param("3.12", marks=pytest.mark.slow),
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
    indirect=True,
)
def test_setup_project(
    project_from_template: Path,
    copier_answers: CopierAnswers,
    template_version_used: str,
    tmp_path: Path,
):
    """Run Copier programmatically to test the the setup for new projects.

    NOTE: This test is slow at first, as it might fill up your UV cache with torch / jax / etc for
    each python version!
    """
    project_root = project_from_template
    # Check that the answers were saved correctly.
    answers_file = project_root / ".copier-answers.yml"
    assert answers_file.exists()
    copier_content = yaml.safe_load(answers_file.read_text())
    for key, value in dataclasses.asdict(copier_answers).items():
        if isinstance(value, list):
            assert sorted(copier_content[key]) == sorted(value)
        else:
            assert copier_content[key] == value

    # Check that tests can be collected without errors. This is usually a good "smoke" test to
    # check for package import errors and such.
    command = ["uv", "run", "pytest", "-v", "--collect-only"]
    logger.info(f"Running: command `{' '.join(command)}`")
    result = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
    if result.returncode != 0:
        print("STDOUT: ", result.stdout)
        print("STDERR: ", result.stderr)
        pytest.fail(f"Failed to collect tests in the new project: {result.stdout}")


class ModifyAndTest(Protocol):
    """
    IDEA:
    1. Start from any of the published versions of the template (including the current (HEAD) version).
    2. Apply some modifications from a list of modifications, e.g. adding a file or something
    3. Yield. If the initial version is not HEAD, `copier update` is run, accepting to overwrite
       anything that is different.
    4. Run a test to check that the project created from the template works as expected.
    """

    def __call__(
        self, project_root: Path, answers: CopierAnswers, template_version_used: str
    ) -> contextlib._GeneratorContextManager[None, None, None]: ...


@contextlib.contextmanager
def add_lit_autoencoder_module(
    project_root: Path, answers: CopierAnswers, template_version_used: str
):
    """Add a module to the project."""
    # todo: This 'module_name' was added after v0.0.1
    project_module = (
        answers.project_name if template_version_used == "v0.0.1" else answers.module_name
    )
    new_module = project_root / project_module / "algorithms" / "lit_autoencoder.py"
    new_module.write_text(
        textwrap.dedent(
            """\
        import os
        from torch import optim, nn, utils, Tensor
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        import lightning as L


        # define the LightningModule
        class LitAutoEncoder(L.LightningModule):
            def __init__(self):
                super().__init__()
                # define any number of nn.Modules (or use your current ones)
                self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
                self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

            def training_step(self, batch, batch_idx):
                # training_step defines the train loop.
                # it is independent of forward
                x, _ = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = nn.functional.mse_loss(x_hat, x)
                # Logging to TensorBoard (if installed) by default
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters(), lr=1e-3)
                return optimizer
        """
        )
    )
    new_algo_config = (
        project_root / project_module / "configs" / "algorithm" / "lit_autoencoder.yaml"
    )
    # Add a Hydra config file for the new module.
    new_algo_config.write_text(
        textwrap.dedent(
            f"""\
        _target_: {project_module}.algorithms.lit_autoencoder.LitAutoEncoder
        """
        )
    )

    yield  # Yield, (let the project update if needed)

    # Then test that it still works:

    command = (
        f"uv run python {answers.module_name}/train.py "
        f"algorithm=lit_autoencoder datamodule=mnist trainer.fast_dev_run=True"
    )

    logger.info(f"Running: command `{command}` in the new, modified project")
    result = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
    if result.returncode != 0:
        print("STDOUT: ", result.stdout)
        print("STDERR: ", result.stderr)
        pytest.fail(f"Failed to collect tests in the new project: {result.stdout}")


# @contextlib.contextmanager
# def no_changes(project_root: Path, answers: CopierAnswers, template_version_used: str):
#     yield


modifications: Sequence[ModifyAndTest] = [
    add_lit_autoencoder_module,
    # no_changes,
]


@pytest.fixture(scope="module")
def version_to(request: pytest.FixtureRequest):
    return getattr(request, "param", "HEAD")


@pytest.mark.skip(reason="TODO: This test is still a work in progress, doesn't exactly work yet.")
@pytest.mark.skipif(
    IN_GITHUB_CLOUD_CI,
    reason="TODO: lots of issues on GitHub CI (commit author, can't install other Python versions).",
)
@pytest.mark.skipif(sys.platform == "win32", reason="The template isn't supported on Windows.")
@pytest.mark.parametrize(
    (template_version_used.__name__, version_to.__name__), sorted(from_to.items()), indirect=True
)
@pytest.mark.parametrize("modification", modifications)
def test_update_project(
    # project_from_template: Path,  # can't use a shared fixture since we modify the project.
    copier_answers: CopierAnswers,
    modification: ModifyAndTest,
    template_version_used: str,
    version_to: str,
    tmp_path: Path,
):
    """IDEA: Test that sets up a project from the template at a given version and then updates it
    to new version with `copier update`.
    """
    project_root = tmp_path / "temp_project"
    data_file = tmp_path / "copier_inputs.yaml"
    data_file.write_text(yaml.dump(dataclasses.asdict(copier_answers)))

    # https://plumbum.readthedocs.io/en/latest/#cheat-sheet
    copier_command = local["uvx"]["copier"]
    copier_command[
        "copy",
        "--trust",
        "--defaults",
        "--skip-answered",
        "--data-file",
        str(data_file),
        f"--vcs-ref={template_version_used}",
        str(REPO_ROOTDIR),
        str(project_root),
    ]()

    with modification(project_root, copier_answers, template_version_used):
        # Need to commit changes for copier to accept to update.
        git = get_git().with_cwd(project_root)
        # TODO: Double-check that these changes only affect the temporary cloned repo, not this
        # current dev repo.
        git("config", "commit.gpgsign", "false")
        local["pre-commit"].with_cwd(project_root)("uninstall")
        git("add", ".")
        git("commit", "-m", f"Apply modifications from {_nameof(modification)}.")

        copier_command.with_cwd(project_root)["update"](
            "--trust", "--defaults", "--vcs-ref", version_to
        )


def _nameof(thing) -> str:
    return getattr(thing, "__name__", str(thing))

# Make a Mock for the remote slurm launcher plugin
# Use monkeypatch.setattr(project.utils.remote_launcher_plugin, ..., that_mock)
# Assert That the mock launcher plugin was instantiated
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import hydra
import hydra.utils
import omegaconf
import pytest
from hydra import initialize_config_module
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher
from milatools.utils.remote_v2 import is_already_logged_in

import project.main
import project.utils.remote_launcher_plugin
from project.configs.config_test import CONFIG_DIR
from project.conftest import command_line_overrides
from project.main import PROJECT_NAME, main
from project.utils import remote_launcher_plugin
from project.utils.remote_launcher_plugin import RemoteSlurmLauncher


def _yaml_files_in(directory: str | Path, recursive: bool = False):
    directory = Path(directory)
    glob = directory.rglob if recursive else directory.glob
    return list(glob("*.yml")) + list(glob("*.yaml"))


cluster_configs = _yaml_files_in(CONFIG_DIR / "cluster")
resource_configs = _yaml_files_in(CONFIG_DIR / "resources")


@pytest.mark.skipif("SLURM_JOB_ID" in os.environ, reason="Can't be run on the cluster just yet.")
@pytest.mark.parametrize(
    command_line_overrides.__name__,
    [
        pytest.param(
            f"algorithm=example datamodule=cifar10 cluster={cluster.stem} resources={resources.stem}",
            marks=pytest.mark.skipif(
                cluster != "mila" and not is_already_logged_in(cluster.stem),
                reason="Logging in would go through 2FA!",
            ),
        )
        for cluster in cluster_configs
        for resources in resource_configs
    ],
    indirect=True,
)
def test_can_load_configs(command_line_arguments: list[str]):
    """Test that the cluster and resource configs can be loaded without errors."""

    with initialize_config_module(
        config_module=f"{PROJECT_NAME}.configs",
        job_name="test",
        version_base="1.2",
    ):
        _config = hydra.compose(
            config_name="config",
            overrides=command_line_arguments,
            return_hydra_config=True,
        )

        launcher_config = _config["hydra"]["launcher"]
        assert launcher_config["_target_"] in [
            remote_launcher_plugin.RemoteSlurmQueueConf._target_,
            SlurmQueueConf._target_,
        ]
        # TODO: Also try to instantiate these configs.

        if launcher_config["_target_"] == remote_launcher_plugin.RemoteSlurmQueueConf._target_:
            with omegaconf.open_dict(launcher_config):
                launcher_config["executor"]["_synced"] = True  # avoid syncing the code here.
            launcher = hydra.utils.instantiate(launcher_config)
            assert isinstance(launcher, remote_launcher_plugin.RemoteSlurmLauncher)
        else:
            launcher = hydra.utils.instantiate(launcher_config)
            assert isinstance(launcher, SlurmLauncher)


in_github_CI = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    in_github_CI,
    # Fails since it will actually sync the code with git push, and a branch isn't checked out.
    reason="Can't be run on the CI",
)
@pytest.mark.skipif("SLURM_JOB_ID" in os.environ, reason="Shouldn't be run on the cluster.")
@pytest.mark.slow()
@pytest.mark.parametrize(
    "argv",
    [
        [
            "algorithm=example",
            "datamodule=cifar10",
            # TODO: The ordering is important here, we can't use `cluster` before `resources`,
            # otherwise it will use the local launcher!
            "resources=gpu",
            "cluster=mila",
            "trainer.fast_dev_run=True",
        ]
    ],
)
def test_run_remote_job(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    launcher_mock = Mock(spec=RemoteSlurmLauncher, wraps=RemoteSlurmLauncher)
    launcher_mock.__qualname__ = RemoteSlurmLauncher.__qualname__
    monkeypatch.setattr(
        project.utils.remote_launcher_plugin, RemoteSlurmLauncher.__name__, launcher_mock
    )

    monkeypatch.setattr(sys, "argv", ["project/main.py"] + argv)
    monkeypatch.setitem(os.environ, "HYDRA_FULL_ERROR", "1")

    main()

    launcher_mock.assert_called_once()

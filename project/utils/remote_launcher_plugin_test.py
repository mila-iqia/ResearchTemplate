# Make a Mock for the remote slurm launcher plugin
# Use monkeypatch.setattr(project.utils.remote_launcher_plugin, ..., that_mock)
# Assert That the mock launcher plugin was instantiated
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_module
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf

import project.main
import project.utils.remote_launcher_plugin
from project.configs.config_test import CONFIG_DIR
from project.main import PROJECT_NAME, main
from project.utils import remote_launcher_plugin
from project.utils.remote_launcher_plugin import RemoteSlurmLauncher


def _yaml_files_in(directory: str | Path, recursive: bool = False):
    directory = Path(directory)
    glob = directory.rglob if recursive else directory.glob
    return list(glob("*.yml")) + list(glob("*.yaml"))


cluster_configs = _yaml_files_in(CONFIG_DIR / "cluster")
resource_configs = _yaml_files_in(CONFIG_DIR / "resources")


@pytest.mark.parametrize(
    "overrides",
    [
        f"algorithm=example datamodule=cifar10 cluster={cluster.stem} resources={resources.stem}"
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
        _config = compose(
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
        # launcher = hydra.utils.instantiate(launcher_config)
        # assert isinstance(launcher, remote_launcher_plugin.RemoteSlurmLauncher | SlurmLauncher)


@pytest.mark.skipif("SLURM_JOB_ID" in os.environ, reason="Shouldn't be run on the cluster.")
@pytest.mark.slow()
@pytest.mark.parametrize(
    "argv",
    [
        [
            "algorithm=example",
            "datamodule=cifar10",
            "cluster=mila",
            "resources=gpu",
            "+trainer.fast_dev_run=True",
        ]
    ],
)
def test_instantiate_remote_slurm_launcher_plugin(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
):
    launcher_mock = Mock(spec=RemoteSlurmLauncher, wraps=RemoteSlurmLauncher)
    monkeypatch.setattr(
        project.utils.remote_launcher_plugin, RemoteSlurmLauncher.__name__, launcher_mock
    )

    monkeypatch.setattr(sys, "argv", ["project/main.py"] + argv)
    monkeypatch.setitem(os.environ, "HYDRA_FULL_ERROR", "1")

    main()

    launcher_mock.assert_called_once()

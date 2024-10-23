# Make a Mock for the remote slurm launcher plugin
# Use monkeypatch.setattr(project.utils.remote_launcher_plugin, ..., that_mock)
# Assert That the mock launcher plugin was instantiated
import os
import sys
from unittest.mock import Mock

import pytest

import project.main
import project.utils.remote_launcher_plugin
from project.main import main
from project.utils.remote_launcher_plugin import RemoteSlurmLauncher


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

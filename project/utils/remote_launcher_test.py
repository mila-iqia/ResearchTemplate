# Make a Mock for the remote slurm launcher plugin
# Use monkeypatch.setattr(project.utils.remote_launcher_plugin, ..., that_mock)
# Assert That the mock launcher plugin was instantiated
import sys
from unittest.mock import Mock

import pytest

import project.main
import project.utils.remote_launcher_plugin
from project.main import main
from project.utils.remote_launcher_plugin import RemoteSlurmLauncher


@pytest.mark.parametrize(
    "argv",
    [
        [
            "algorithm=example",
            "datamodule=cifar10",
            "cluster=debug",
            "resources=one_gpu",
            "+trainer.fast_dev_run=True",
        ]
    ],
)
# @pytest.mark.slow() add integration tests based on mocking remote_launcher methods
def test_instantiate_remote_slurm_launcher_plugin(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
):
    print(argv)
    launcher_mock = Mock(wraps=RemoteSlurmLauncher)
    monkeypatch.setattr(
        project.utils.remote_launcher_plugin, RemoteSlurmLauncher.__name__, launcher_mock
    )

    monkeypatch.setattr(sys, "argv", ["project/main.py"] + argv)
    main()

    launcher_mock.assert_called_once()
    # assert False, launcher_mock.method_calls
    launcher_mock.submit.assert_called_once()
    assert launcher_mock.call_args_list[0].kwargs["cluster_hostname"] == "mila"

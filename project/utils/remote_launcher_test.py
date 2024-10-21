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
    [["algorithm=example", "datamodule=cifar10", "cluster=mila", "resources=one_gpu"]],
)
def test_instantiate_remote_slurm_launcher_plugin(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
):
    print(argv)
    launcher_mock = Mock(wraps=RemoteSlurmLauncher)
    monkeypatch.setattr(
        project.utils.remote_launcher_plugin, RemoteSlurmLauncher.__name__, launcher_mock
    )

    monkeypatch.setattr(sys, "argv", ["project/main.py"] + argv)
    result = main()


#    assert launcher_mock.called

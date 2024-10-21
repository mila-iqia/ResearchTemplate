# Make a Mock for the remote slurm launcher plugin
# Use monkeypatch.setattr(project.utils.remote_launcher_plugin, ..., that_mock)
# Assert That the mock launcher plugin was instantiated
from unittest.mock import Mock

import pytest

from project.utils.remote_launcher_plugin import RemoteSlurmLauncher


def test_instantiate_remote_slurm_launcher_plugin(monkeypatch: pytest.MonkeyPatch):
    launcher_mock = Mock(spec=RemoteSlurmLauncher)
    monkeypatch.setattr("project.utils.remote_launcher_plugin.RemoteSlurmLauncher", launcher_mock)
    launcher_mock(cluster_hostname="mila")
    launcher_mock.assert_called_once_with(
        cluster_hostname="mila",
    )

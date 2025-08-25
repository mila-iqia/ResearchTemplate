"""Tests for the submitit launcher plugin that allows packing more than one "run" per job."""

import os
import shlex
import sys
from unittest.mock import Mock

import hydra
import hydra.utils
import omegaconf
import pytest
import torch
from hydra import initialize_config_module
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher

import project.main
import project.utils.packing_launcher_plugin
import project.utils.remote_launcher_plugin
from project.main import PROJECT_NAME, main
from project.utils import remote_launcher_plugin
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.packing_launcher_plugin import PackedSlurmLauncher


@pytest.mark.parametrize(
    "command_line_args",
    [
        pytest.param(
            "algorithm=image_classifier datamodule=cifar10 trainer.fast_dev_run=True cluster=current "
            "resources=packed_gpu trainer.max_epochs=1,2,3",
            marks=[
                pytest.mark.skipif(
                    SLURM_JOB_ID is None or not torch.cuda.is_available(),
                    reason="Can only be run on a compute node with a GPU on a slurm cluster.",
                )
            ],
        )
    ],
)
def test_can_load_configs(command_line_args: str):
    """Test that the cluster and resource configs can be loaded without errors."""

    with initialize_config_module(
        config_module=f"{PROJECT_NAME}.configs",
        job_name="test",
        version_base="1.2",
    ):
        overrides = shlex.split(command_line_args)
        _config = hydra.compose(
            config_name="config",
            overrides=overrides,
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
            # TODO: This still tries to `git push`, which fails on the CI.
            # launcher = hydra.utils.instantiate(launcher_config)
            # assert isinstance(launcher, remote_launcher_plugin.RemoteSlurmLauncher)
        else:
            launcher = hydra.utils.instantiate(launcher_config)
            assert isinstance(launcher, SlurmLauncher)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            [
                "algorithm=image_classifier",
                "datamodule=cifar10",
                "trainer.limit_train_batches=0.1",
                "cluster=current",
                "resources=packed_gpu",
                "trainer.max_epochs=1,2,3",
            ],
            marks=[
                pytest.mark.skipif(
                    SLURM_JOB_ID is None or not torch.cuda.is_available(),
                    reason="Can only be run on a compute node with a GPU on a slurm cluster.",
                )
            ],
        )
    ],
)
def test_run_remote_job(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    launcher_mock = Mock(spec=PackedSlurmLauncher, wraps=PackedSlurmLauncher)
    launcher_mock.__qualname__ = PackedSlurmLauncher.__qualname__
    monkeypatch.setattr(
        project.utils.packing_launcher_plugin, PackedSlurmLauncher.__name__, launcher_mock
    )
    monkeypatch.setattr(sys, "argv", ["project/main.py"] + argv)
    monkeypatch.setitem(os.environ, "HYDRA_FULL_ERROR", "1")

    main()

    launcher_mock.assert_called_once()
    assert False, launcher_mock.return_value

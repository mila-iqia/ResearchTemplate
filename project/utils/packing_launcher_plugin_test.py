"""Tests for the submitit launcher plugin that allows packing more than one "run" per job."""

import os
import sys
from unittest.mock import Mock

import pytest
import torch

import project.main
import project.utils.packing_launcher_plugin
import project.utils.remote_launcher_plugin
from project.main import main
from project.utils.env_vars import SLURM_JOB_ID
from project.utils.packing_launcher_plugin import PackedSlurmLauncher


@pytest.mark.slow
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            [
                "--multirun",
                "algorithm=image_classifier",
                "datamodule=cifar10",
                # "cluster=current",
                "resources=packed_gpu",
                "hydra.launcher.additional_parameters.time=00:10:00",
                "hydra.launcher.tasks_per_node=3",  # feels dumb to have to set this! Shouldn't it default to the number of combinations to run?
                "+trainer.limit_train_batches=0.1",
                "trainer.max_epochs=1,2,3",  # Hopefully see three results with increasing accuracy
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

    _results_of_main = main()
    assert _results_of_main is None
    # NOTE: for some reason Hydra decided that main with multirun should return nothing?!
    # Why not return the list of results?
    assert isinstance(launcher_mock, PackedSlurmLauncher)
    launcher_mock.assert_called_once()
    assert False, launcher_mock.launch.return_value

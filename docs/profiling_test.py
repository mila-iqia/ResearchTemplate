import shutil

import pytest
from omegaconf import DictConfig

from project.conftest import (  # noqa: F401
    accelerator,
    algorithm_config,
    command_line_arguments,
    datamodule_config,
    devices,
    experiment_dictconfig,
    network_config,
    num_devices_to_use,
    overrides,
)
from project.experiment import setup_experiment
from project.utils.hydra_utils import resolve_dictconfig

## Direct parametrization would occur with the following code
# @pytest.fixture
# def datamodule(command_line_arguments: str) -> LightningDataModule:
#     # uses hydra to get the datamodule from the command line arguments
#     # Something like:
#     # hydra.utils.instantiate(config=command_line_arguments)
#     # with hydra.utils.setup
#     ...


# @pytest.mark.parametrize("foo", [1, 2, 3])
# def test_something_about_datamodule(datamodule: LightningDataModule, foo: int):
#     datamodule.prepare_data()
#     datamodule.setup("fit")
#     ...


@pytest.mark.skipif(not shutil.which("sbatch"), reason="Needs to be run on a SLURM cluster")
@pytest.mark.parametrize(
    "command_line_arguments",
    [
        # Instrumenting your code -baseline
        """
        experiment=profiling \
        trainer.logger.wandb.name="Baseline" \
        trainer.logger.wandb.tags=["Training","Baseline comparison","CPU/GPU comparison"]
        """,
        # Identifying potential bottlenecks - baseline
        """
        experiment=profiling \
        algorithm=no_op\
        trainer.logger.wandb.name="Baseline without training" \
        trainer.logger.wandb.tags=["No training","Baseline comparison"]
        """,
        # Identifying potential bottlenecks - num_workers multirun
        """
        -m experiment=profiling \
        algorithm=no_op \
        trainer.logger.wandb.tags=["1 CPU Dataloading","Worker throughput"] \
        datamodule.num_workers=1,4,8,16,3"
        """,
        # Identifying potential bottlenecks - num_workers multirun
        """
        experiment=profiling \
        algorithm=no_op \
        resources=cpu \
        trainer.logger.wandb.tags=["2 CPU Dataloading","Worker throughput"] \
        hydra.launcher.timeout_min=60 \
        hydra.launcher.cpus_per_task=2 \
        hydra.launcher.constraint="sapphire" \
        datamodule.num_workers=1,4,8,16,32``
        """,
        # Identifying potential bottlenecks - fcnet mnist
        """
        experiment=profiling \
        network=fcnet \
        datamodule=mnist \
        trainer.logger.wandb.name="FcNet/MNIST baseline with training" \
        trainer.logger.wandb.tags=["CPU/GPU comparison","GPU","MNIST"]
        """,
        # Throughput across GPU types
        """
        experiment=profiling \
        resources=one_gpu \
        hydra.launcher.gres='gpu:a100:1' \
        hydra.launcher.cpus_per_task=4 \
        datamodule.num_workers=8 \
        trainer.logger.wandb.name="A100 training" \
        trainer.logger.wandb.tags=["GPU comparison"]
        """,
        # Making the most out of your GPU
        """
        experiment=profiling \
        datamodule.num_workers=8 \
        datamodule.batch_size=32,64,128,256,512 \
        trainer.logger.wandb.tags=["Batch size comparison"]\
        '++trainer.logger.wandb.name=Batch size ${datamodule.batch_size}'
        """,
    ],
    indirect=True,
)
def test_notebook_commands_dont_cause_errors(experiment_dictconfig: DictConfig):  # noqa
    # check for any errors related to OmegaConf interpolations and such
    config = resolve_dictconfig(experiment_dictconfig)
    # check for any errors when actually instantiating the components.
    _experiment = setup_experiment(config)
    # Note: Here we don't actually do anything with the objects.

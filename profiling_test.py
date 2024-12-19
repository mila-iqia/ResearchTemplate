import shutil

import hydra.errors
import lightning
import pytest
from omegaconf import DictConfig

from project.conftest import (  # noqa: F401
    accelerator,
    algorithm_config,
    algorithm_network_config,
    command_line_arguments,
    command_line_overrides,
    datamodule_config,
    experiment_dictconfig,
)
from project.experiment import instantiate_datamodule, instantiate_trainer
from project.main import (
    instantiate_algorithm,
    setup_logging,
)
from project.utils.hydra_utils import resolve_dictconfig


# NTOE: could also run these commands with the `resources` group and `cluster=mila`
@pytest.mark.skipif(not shutil.which("sbatch"), reason="Needs to be run on a SLURM cluster")
@pytest.mark.parametrize(
    "command_line_arguments",
    [
        # Instrumenting your code -baseline
        """
        experiment=profiling \
        algorithm=image_classifier \
        trainer.logger.wandb.name="Baseline" \
        trainer.logger.wandb.tags=["Training","Baseline comparison","CPU/GPU comparison"]
        """,
        # Identifying potential bottlenecks - baseline
        """
        experiment=profiling\
        algorithm=no_op\
        trainer.logger.wandb.name="Baseline without training" \
        trainer.logger.wandb.tags=["No training","Baseline comparison"]

        """,
        # Identifying potential bottlenecks - num_workers multirun
        pytest.param(
            """
            -m experiment=profiling \
            algorithm=no_op \
            trainer.logger.wandb.tags=["1 CPU Dataloading","Worker throughput"] \
            datamodule.num_workers=1,4,8,16,32
            """,
            marks=pytest.mark.xfail(
                reason="LexerNoViableAltException error caused by the -m flag",
                raises=hydra.errors.OverrideParseException,
                strict=True,
            ),
        ),
        # Identifying potential bottlenecks - num_workers multirun
        pytest.param(
            """
        -m experiment=profiling \
        algorithm=no_op \
        resources=cpu \
        trainer.logger.wandb.tags=["2 CPU Dataloading","Worker throughput"] \
        hydra.launcher.timeout_min=60 \
        hydra.launcher.cpus_per_task=2 \
        hydra.launcher.constraint="sapphire" \
        datamodule.num_workers=1,4,8,16,32
        """,
            marks=pytest.mark.xfail(
                reason="LexerNoViableAltException error caused by the -m flag",
                raises=hydra.errors.OverrideParseException,
                strict=True,
            ),
        ),
        # Identifying potential bottlenecks - fcnet mnist
        """
        experiment=profiling \
        algorithm=image_classifier \
        algorithm/network=fcnet \
        datamodule=mnist \
        trainer.logger.wandb.name="FcNet/MNIST baseline with training" \
        trainer.logger.wandb.tags=["CPU/GPU comparison","GPU","MNIST"]
        """,
        # Throughput across GPU types
        """
        experiment=profiling \
        algorithm=image_classifier \
        resources=gpu \
        hydra.launcher.gres='gpu:a100:1' \
        hydra.launcher.cpus_per_task=4 \
        datamodule.num_workers=8 \
        trainer.logger.wandb.name="A100 training" \
        trainer.logger.wandb.tags=["GPU comparison"]
        """,
        # Making the most out of your GPU
        pytest.param(
            """
        -m experiment=profiling \
        algorithm=image_classifier \
        datamodule.num_workers=8 \
        datamodule.batch_size=32,64,128,256 \
        trainer.logger.wandb.tags=["Batch size comparison"]\
        '++trainer.logger.wandb.name=Batch size ${datamodule.batch_size}'
        """,
            marks=pytest.mark.xfail(
                reason="LexerNoViableAltException error caused by the -m flag",
                raises=hydra.errors.OverrideParseException,
                strict=True,
            ),
        ),
    ],
    indirect=True,
)
def test_notebook_commands_dont_cause_errors(experiment_dictconfig: DictConfig):  # noqa
    # check for any errors related to OmegaConf interpolations and such
    config = resolve_dictconfig(experiment_dictconfig)
    # check for any errors when actually instantiating the components.
    # _experiment = _setup_experiment(config)
    setup_logging(log_level=config.log_level)
    lightning.seed_everything(config.seed, workers=True)
    _trainer = instantiate_trainer(config.trainer)
    datamodule = instantiate_datamodule(config.datamodule)
    _algorithm = instantiate_algorithm(config, datamodule=datamodule)

    # Note: Here we don't actually do anything with the objects.

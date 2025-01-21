"""Training script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm), optionally datamodule and network;
3. Trains the model;
4. Optionally runs an evaluation loop.

"""

from __future__ import annotations

import functools
import logging
import os
import typing
from pathlib import Path

import hydra
import lightning
import omegaconf
import rich
import rich.logging
import wandb
from hydra_plugins.auto_schema import auto_schema_plugin
from omegaconf import DictConfig

import project
from project.configs import add_configs_to_hydra_store
from project.configs.config import Config
from project.experiment import evaluate, instantiate_datamodule, instantiate_trainer, train
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.utils import print_config

if typing.TYPE_CHECKING:
    from project.trainers.jax_trainer import JaxModule

PROJECT_NAME = project.__name__
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


auto_schema_plugin.config = auto_schema_plugin.AutoSchemaPluginConfig(
    schemas_dir=REPO_ROOTDIR / ".schemas",
    regen_schemas=False,
    stop_on_error=False,
    quiet=False,
    verbose=False,
    add_headers=False,  # don't fallback to adding headers if we can't use vscode settings file.
)

add_configs_to_hydra_store()


@hydra.main(
    config_path=f"pkg://{PROJECT_NAME}.configs",
    config_name="config",
    version_base="1.2",
)
def main(dict_config: DictConfig) -> dict:
    """Main entry point for training a model.

    This does roughly the same thing as
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

    1. Instantiates the experiment components from the Hydra configuration:
        - trainer
        - algorithm
        - datamodule (optional)
    2. Calls `train` to train the algorithm
    3. Calls `evaluation` to evaluate the model
    4. Returns the evaluation metrics.
    """

    print_config(dict_config, resolve=False)
    assert dict_config["algorithm"] is not None

    # Resolve all the interpolations in the configs.
    config: Config = resolve_dictconfig(dict_config)
    setup_logging(
        log_level=config.log_level,
        global_log_level="DEBUG" if config.debug else "INFO" if config.verbose else "WARNING",
    )

    # Seed the random number generators, so the weights that are
    # constructed are deterministic and reproducible.
    lightning.seed_everything(seed=config.seed, workers=True)

    # Create the algo.
    algorithm = instantiate_algorithm(config)

    # Create the trainer
    trainer = instantiate_trainer(config.trainer)

    if wandb.run:
        wandb.run.config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
        wandb.run.config.update(
            omegaconf.OmegaConf.to_container(dict_config, resolve=False, throw_on_missing=True)
        )

    # Train the algorithm.
    algorithm, train_results = train(
        algorithm,
        trainer=trainer,
        config=config,
    )

    # Evaluate the algorithm.
    metric_name, error, _metrics = evaluate(
        algorithm,
        trainer=trainer,
        train_results=train_results,
        config=config,
    )

    if wandb.run:
        wandb.finish()

    assert error is not None
    # Results are returned like this so that the Orion sweeper can parse the results correctly.
    return dict(name=metric_name, type="objective", value=error)


def setup_logging(log_level: str, global_log_level: str = "WARNING") -> None:
    from project.main import PROJECT_NAME

    logging.basicConfig(
        level=global_log_level.upper(),
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    project_logger = logging.getLogger(PROJECT_NAME)
    project_logger.setLevel(log_level.upper())


def instantiate_algorithm(
    config: Config, datamodule: lightning.LightningDataModule | None = None
) -> lightning.LightningModule | JaxModule:
    """Function used to instantiate the algorithm.

    It is suggested that your algorithm (LightningModule) take in the `datamodule` and `network`
    as arguments, to make it easier to swap out different networks and datamodules during
    experiments.

    The instantiated datamodule and network will be passed to the algorithm's constructor.
    """

    # Create the algorithm
    algo_config = config.algorithm

    # Create the datamodule (if present) from the config
    if datamodule is None and config.datamodule is not None:
        datamodule = instantiate_datamodule(config.datamodule)

    if datamodule:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config, datamodule=datamodule)
    else:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config)

    if isinstance(algo_or_algo_partial, functools.partial):
        if datamodule:
            return algo_or_algo_partial(datamodule=datamodule)
        return algo_or_algo_partial()

    algorithm = algo_or_algo_partial
    return algorithm


if __name__ == "__main__":
    main()

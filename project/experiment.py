from __future__ import annotations

import logging
from dataclasses import dataclass, is_dataclass
from logging import getLogger as get_logger

from hydra_zen import instantiate
from lightning import Callback, LightningDataModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from project.algorithms import Algorithm
from project.configs.config import Config
from project.datamodules.datamodule import DataModule
from lightning.pytorch.loggers import Logger
from project.utils.hydra_utils import get_outer_class

from project.utils.utils import validate_datamodule
from torch import nn


logger = get_logger(__name__)


@dataclass
class Experiment:
    """Dataclass containing everything used in an experiment.

    This gets created from the config that are parsed from Hydra. Can be used to run the experiment
    by calling `run(experiment)`. Could also be serialized to a file or saved to disk, which might
    come in handy with `submitit` later on.
    """

    algorithm: Algorithm
    network: nn.Module
    datamodule: DataModule
    trainer: Trainer


def setup_datamodule(experiment_config: Config) -> LightningDataModule:
    """Sets up the datamodule from the config.

    This has a few differences w.r.t. just doing `instantiate(experiment_config.datamodule)`:
    1. If there is a `batch_size` attribute on the Algorithm's HParams, sets that attribute on the
       datamodule.
    2. If the datamodule is a VisionDataModule and has `normalize=False`, removes any normalization
    transform from the train and val transforms, if present attribute.
    """
    if isinstance(experiment_config.datamodule, LightningDataModule):
        return experiment_config.datamodule

    datamodule_config = experiment_config.datamodule
    if isinstance(datamodule_config, (dict, DictConfig)):
        assert "_target_" in datamodule_config
        # datamodule = OmegaConf.to_object(datamodule_config)
        datamodule = instantiate(datamodule_config)
        assert isinstance(datamodule, LightningDataModule)
        return datamodule

    if not is_dataclass(datamodule_config):
        raise NotImplementedError(
            f"Assuming that the 'datamodule' config entry is either a datamodule or a config "
            f"dataclass. (Got {datamodule_config} of type {type(datamodule_config)})."
        )

    datamodule_overrides = {}
    if hasattr(experiment_config.algorithm, "batch_size"):
        # The algorithm has the batch size as a hyper-parameter.
        algo_batch_size = getattr(experiment_config.algorithm, "batch_size")
        assert isinstance(algo_batch_size, int)
        logger.info(
            f"Overwriting `batch_size` from datamodule config with the value on the Algorithm "
            f"hyper-parameters: {algo_batch_size}"
        )
        datamodule_overrides["batch_size"] = algo_batch_size
    datamodule = instantiate(datamodule_config, **datamodule_overrides)
    datamodule = validate_datamodule(datamodule)
    return datamodule


def setup_experiment(experiment_config: Config) -> Experiment:
    """Do all the postprocessing necessary (e.g., create the network, Algorithm, datamodule,
    callbacks, Trainer, etc) to go from the options that come from Hydra, into all required
    components for the experiment, which is stored as a namedtuple-like class called `Experiment`.

    NOTE: This also has the effect of seeding the random number generators, so the weights that are
    constructed are always deterministic.
    """

    root_logger = logging.getLogger("project")
    root_logger.setLevel(getattr(logging, experiment_config.log_level.upper()))
    logger.info(f"Using random seed: {experiment_config.seed}")
    seed_everything(seed=experiment_config.seed, workers=True)

    # Create the Trainer.
    trainer_config = experiment_config.trainer.copy()
    if isinstance(trainer_config, DictConfig):
        trainer_config = OmegaConf.to_container(trainer_config)
    assert isinstance(trainer_config, dict)
    # NOTE: The callbacks and loggers are parsed into dict[str, obj] by Hydra, but we need them
    # to be passed as a list of objects to the Trainer. Therefore here we put the dicts in lists
    callbacks: dict[str, Callback] = instantiate(trainer_config.pop("callbacks", {})) or {}
    loggers: dict[str, Logger] = instantiate(trainer_config.pop("logger", {})) or {}
    trainer = instantiate(
        trainer_config,
        callbacks=list(callbacks.values()),
        logger=list(loggers.values()),
    )
    assert isinstance(trainer, Trainer)

    # Create the datamodule:
    # datamodule = instantiate(experiment_config.datamodule)
    datamodule = setup_datamodule(experiment_config)

    # TODO: Seems a bit too rigid to have the network be create independently of the algorithm.
    # This might need to change.
    # Create the network
    network = instantiate(experiment_config.network)
    assert isinstance(network, nn.Module), (network, type(network))

    # Create the algorithm
    if isinstance(experiment_config.algorithm, Algorithm.HParams):
        algo_hparams = experiment_config.algorithm
        algo_type: type[Algorithm] = getattr(
            algo_hparams, "_target_", get_outer_class(type(algo_hparams))
        )
        algorithm = algo_type(
            datamodule=datamodule,
            network=network,
            hp=algo_hparams,
        )
    else:
        algorithm = instantiate(
            experiment_config.algorithm,
            datamodule=datamodule,
            network=network,
        )
    assert isinstance(algorithm, Algorithm), (algorithm, type(algorithm))

    return Experiment(
        trainer=trainer,
        algorithm=algorithm,
        network=network,
        datamodule=datamodule,
    )

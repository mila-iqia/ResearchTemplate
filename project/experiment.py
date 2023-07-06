from __future__ import annotations

import logging
import random
from dataclasses import dataclass, is_dataclass
from logging import getLogger as get_logger

from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from project.algorithms import Algorithm
from project.configs.config import Config
from project.datamodules.datamodule import DataModule
from project.utils.hydra_utils import get_outer_class
from lightning.pytorch.loggers import Logger

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
    if isinstance(experiment_config.datamodule, LightningDataModule):
        return experiment_config.datamodule

    datamodule_config = experiment_config.datamodule
    if isinstance(datamodule_config, (dict, DictConfig)):
        assert "_target_" in datamodule_config
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

    root_logger = logging.getLogger()
    if experiment_config.debug:
        root_logger.setLevel(logging.INFO)
    elif experiment_config.verbose:
        root_logger.setLevel(logging.DEBUG)

    if experiment_config.seed is not None:
        seed = experiment_config.seed
        print(f"seed manually set to {experiment_config.seed}")
    else:
        seed = random.randint(0, int(1e5))
        print(f"Randomly selected seed: {seed}")
    seed_everything(seed=seed, workers=True)

    # Create the datamodule:
    datamodule = setup_datamodule(experiment_config)

    # Create the Trainer.
    # NOTE: The callbacks and loggers are parsed into dict[str, obj] by Hydra, but we need them
    # to be passed as a list of objects to the Trainer. Therefore here we put the dicts in lists
    callbacks: dict[str, dict] = experiment_config.trainer.get("callbacks", {}) or {}
    loggers: dict[str, dict] = experiment_config.trainer.setdefault("logger", {}) or {}
    trainer = instantiate(
        experiment_config.trainer,
        callbacks=list(experiment_config.trainer.setdefault("callbacks", {}).values()),
        logger=list(experiment_config.trainer.setdefault("logger", {}).values()),
    )
    assert isinstance(trainer, Trainer)

    # Create the network
    # network = instantiate(experiment_config.network)
    # assert isinstance(network, nn.Module), network
    # Create the algorithm
    # TODO: Seems a bit too rigid to have the network be create independently of the algorithm.
    # This might need to change.
    # assert False, experiment_config.algorithm
    # algorithm = instantiate(experiment_config.algorithm, network=132, datamodule=456)
    experiment_config.algorithm["datamodule"] = datamodule
    assert False, experiment_config.network
    algorithm = instantiate(
        experiment_config.algorithm,
        datamodule=datamodule,
        network=experiment_config.network,
    )
    # algo_hparams: Algorithm.HParams = experiment_config.algorithm
    # algorithm_type: type[Algorithm] = get_outer_class(type(algo_hparams))
    # assert isinstance(
    #     algo_hparams, algorithm_type.HParams  # type: ignore
    # ), "HParams type should match model type"

    # algorithm = algorithm_type(
    #     datamodule=datamodule,
    #     network=network,
    #     hp=algo_hparams,
    # )

    return Experiment(
        trainer=trainer,
        algorithm=algorithm,
        network=network,
        datamodule=datamodule,
    )

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, is_dataclass
from logging import getLogger as get_logger

from hydra_zen import instantiate
from lightning import Callback, LightningDataModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from project.algorithms import Algorithm
from project.configs.config import Config
from project.datamodules.datamodule import DataModule
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
    root_logger.setLevel(getattr(logging, experiment_config.log_level.upper()))

    if experiment_config.seed is not None:
        seed = experiment_config.seed
        print(f"seed manually set to {experiment_config.seed}")
    else:
        seed = random.randint(0, int(1e5))
        print(f"Randomly selected seed: {seed}")
    seed_everything(seed=seed, workers=True)

    # Create the datamodule:
    datamodule = instantiate(experiment_config.datamodule)
    # Create the network

    # NOTE: Here we register a way to get the attribute of the *object* instead of its config:
    OmegaConf.register_new_resolver("datamodule", datamodule.__getattribute__)

    # NOTE: Doesn't work, because it needs to have access to the datamodule *instance* attributes!
    # Could we perhaps create a function that provides more "context variables" for the
    # interpolation?
    network = instantiate(experiment_config.network)

    # Create the Trainer.
    trainer_config = OmegaConf.to_container(experiment_config.trainer)
    assert isinstance(trainer_config, dict)
    # NOTE: The callbacks and loggers are parsed into dict[str, obj] by Hydra, but we need them
    # to be passed as a list of objects to the Trainer. Therefore here we put the dicts in lists
    callbacks: dict[str, Callback] = instantiate(trainer_config.pop("callbacks", {})) or {}
    loggers: dict[str, Logger] = instantiate(trainer_config.pop("logger", {})) or {}

    trainer = instantiate(
        trainer_config,
        callbacks=list(callbacks.values()),
        logger=list(loggers.values()),
        # callbacks="${oc.dict.values: /trainer/callbacks}",
        # logger="${oc.dict.values: /trainer/logger}",
    )
    assert isinstance(trainer, Trainer)

    # Create the network
    # network = instantiate(experiment_config.network)
    # TODO: Shouldn't we let the algorithm do it though?
    # assert isinstance(network, (nn.Module, functools.partial))
    # if isinstance(network, functools.partial):
    #     network = network()
    # Create the algorithm
    # TODO: Seems a bit too rigid to have the network be create independently of the algorithm.
    # This might need to change.
    # algorithm = instantiate(experiment_config.algorithm, network=132, datamodule=456)
    # experiment_config.algorithm["datamodule"] = datamodule
    algorithm = instantiate(
        experiment_config.algorithm,
        datamodule=datamodule,
        network=network,
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

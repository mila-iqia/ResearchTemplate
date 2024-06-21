from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, is_dataclass
from logging import getLogger as get_logger
from typing import Any

import hydra_zen
import rich.console
import rich.logging
import rich.traceback
import torch
from lightning import Callback, Trainer, seed_everything
from omegaconf import DictConfig
from torch import nn

from project.algorithms import Algorithm
from project.configs.config import Config
from project.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from project.utils.hydra_utils import get_outer_class, instantiate
from project.utils.types import Dataclass
from project.utils.types.protocols import DataModule, Module
from project.utils.utils import validate_datamodule

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


def setup_experiment(experiment_config: Config) -> Experiment:
    """Do all the postprocessing necessary (e.g., create the network, datamodule, callbacks,
    Trainer, Algorithm, etc) to go from the options that come from Hydra, into all required
    components for the experiment, which is stored as a dataclass called `Experiment`.

    NOTE: This also has the effect of seeding the random number generators, so the weights that are
    constructed are deterministic and reproducible.
    """
    setup_logging(experiment_config)
    seed_rng(experiment_config)
    trainer = instantiate_trainer(experiment_config)

    datamodule = instantiate_datamodule(experiment_config)

    network = instantiate_network(experiment_config, datamodule=datamodule)

    algorithm = instantiate_algorithm(experiment_config, datamodule=datamodule, network=network)

    return Experiment(
        trainer=trainer,
        algorithm=algorithm,
        network=network,  # todo: fix typing issues (maybe removing/reworking the `Network` class?)
        datamodule=datamodule,
    )


def setup_logging(experiment_config: Config) -> None:
    LOGLEVEL = os.environ.get("LOGLEVEL", "info").upper()
    logging.basicConfig(
        level=LOGLEVEL,
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

    root_logger = logging.getLogger("project")

    if experiment_config.debug:
        root_logger.setLevel(logging.INFO)
    elif experiment_config.verbose:
        root_logger.setLevel(logging.DEBUG)


def seed_rng(experiment_config: Config):
    if experiment_config.seed is not None:
        seed = experiment_config.seed
        print(f"seed manually set to {experiment_config.seed}")
    else:
        seed = random.randint(0, int(1e5))
        print(f"Randomly selected seed: {seed}")
    seed_everything(seed=seed, workers=True)


def instantiate_trainer(experiment_config: Config) -> Trainer:
    # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
    # fields have the right type.

    # instantiate all the callbacks
    callback_configs = experiment_config.trainer.pop("callbacks", {})
    callback_configs = {k: v for k, v in callback_configs.items() if v is not None}
    callbacks: dict[str, Callback] | None = instantiate(callback_configs)
    # Create the loggers, if any.
    loggers: dict[str, Any] | None = instantiate(experiment_config.trainer.pop("logger", {}))
    # Create the Trainer.
    assert isinstance(experiment_config.trainer, dict)
    if experiment_config.debug:
        logger.info("Setting the max_epochs to 1, since the 'debug' flag was passed.")
        experiment_config.trainer["max_epochs"] = 1
    if "_target_" not in experiment_config.trainer:
        experiment_config.trainer["_target_"] = Trainer

    trainer = instantiate(
        experiment_config.trainer,
        callbacks=list(callbacks.values()) if callbacks else None,
        logger=list(loggers.values()) if loggers else None,
    )
    assert isinstance(trainer, Trainer)
    return trainer


def instantiate_datamodule(experiment_config: Config) -> DataModule:
    datamodule_config: Dataclass | DataModule = experiment_config.datamodule

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

    datamodule: DataModule
    if isinstance(datamodule_config, DataModule):
        logger.info(
            f"Datamodule was already instantiated (probably to interpolate a field value). "
            f"{datamodule_config=}"
        )
        datamodule = datamodule_config
    else:
        datamodule = instantiate(datamodule_config, **datamodule_overrides)
        assert isinstance(datamodule, DataModule)

    datamodule = validate_datamodule(datamodule)
    return datamodule


def get_experiment_device(experiment_config: Config | DictConfig) -> torch.device:
    if experiment_config.trainer.get("accelerator", "cpu") == "gpu":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def instantiate_network(experiment_config: Config, datamodule: DataModule) -> nn.Module:
    device = get_experiment_device(experiment_config)

    network_config = experiment_config.network

    # todo: Should we wrap flax.linen.Modules into torch modules automatically for torch-based algos?

    if isinstance(network_config, dict | DictConfig) or hasattr(network_config, "_target_"):
        with device:
            network = hydra_zen.instantiate(network_config)
    elif is_dataclass(network_config):
        with device:
            network = instantiate_network_from_hparams(
                network_hparams=network_config, datamodule=datamodule
            )
        assert isinstance(network, nn.Module)
    elif isinstance(network_config, nn.Module):
        logger.warning(
            RuntimeWarning(
                f"The network config is a nn.Module. Consider using a _target_ or _partial_"
                f"in a config instead, so the config stays lightweight. (network={network_config})"
            )
        )
        network = network_config.to(device=device)
    elif callable(network_config):
        # for example when using _partial_ in a config.
        with device:
            network = network_config()
    else:
        raise RuntimeError(f"Unsupported network config passed: {network_config}")

    return network


def instantiate_algorithm(
    experiment_config: Config, datamodule: DataModule, network: nn.Module
) -> Algorithm:
    # Create the algorithm
    algo_config = experiment_config.algorithm
    if isinstance(algo_config, Algorithm):
        logger.info(
            f"Algorithm was already instantiated (probably to interpolate a field value)."
            f"{algo_config=}"
        )
        return algo_config

    if isinstance(algo_config, dict | DictConfig):
        if "_target_" not in algo_config:
            raise NotImplementedError(
                "The algorithm config, if a dict, should have a _target_ set to an Algorithm class."
            )
        if algo_config.get("_partial_", False):
            algo_config = instantiate(algo_config)
            algorithm = algo_config(datamodule=datamodule, network=network)
        else:
            algorithm = instantiate(algo_config, datamodule=datamodule, network=network)

        if not isinstance(algorithm, Algorithm):
            raise NotImplementedError(
                f"The algorithm config didn't create an Algorithm instance:\n"
                f"{algo_config=}\n"
                f"{algorithm=}"
            )
        return algorithm

    if hasattr(algo_config, "_target_"):
        # A dataclass of some sort, with a _target_ attribute.
        algorithm = instantiate(algo_config, datamodule=datamodule, network=network)
        assert isinstance(algorithm, Algorithm)
        return algorithm

    if not isinstance(algo_config, Algorithm.HParams):
        raise NotImplementedError(
            f"For now the algorithm config can either have a _target_ set to an Algorithm class, "
            f"or configure an inner Algorithm.HParams dataclass. Got:\n{algo_config=}"
        )

    algorithm_type: type[Algorithm] = get_outer_class(type(algo_config))
    assert isinstance(
        algo_config,
        algorithm_type.HParams,  # type: ignore
    ), "HParams type should match model type"

    algorithm = algorithm_type(
        datamodule=datamodule,
        network=network,
        hp=algo_config,
    )
    return algorithm


def instantiate_network_from_hparams(network_hparams: Dataclass, datamodule: DataModule) -> Module:
    """TODO: Refactor this if possible. Shouldn't be as complicated as it currently is.

    Perhaps we could register handler functions for each pair of datamodule and network type, a bit
    like a multiple dispatch?
    """
    network_type = get_outer_class(type(network_hparams))
    assert issubclass(network_type, nn.Module)
    assert isinstance(
        network_hparams,
        network_type.HParams,  # type: ignore
    ), "HParams type should match net type"
    if isinstance(datamodule, ImageClassificationDataModule):
        # if issubclass(network_type, ImageClassifierNetwork):
        return network_type(
            in_channels=datamodule.dims[0],
            n_classes=datamodule.num_classes,  # type: ignore
            hparams=network_hparams,
        )

    raise NotImplementedError(datamodule, network_hparams)

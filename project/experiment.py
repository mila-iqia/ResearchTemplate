"""Module containing the functions which create experiment components from Hydra configs.

This is essentially just calling [hydra.utils.instantiate](
https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/#internaldocs-banner)
on the
datamodule, network, trainer, and algorithm configs in a certain order.

This also adds the instance_attr custom resolver, which allows you to retrieve an attribute of
an instantiated object instead of a config.
"""

from __future__ import annotations

import copy
import functools
import logging
import typing
from typing import Any

import hydra
import hydra.utils
import hydra_zen
import rich.console
import rich.logging
import rich.traceback

if typing.TYPE_CHECKING:
    from hydra_zen.typing import Builds
    from lightning import Callback, LightningDataModule, LightningModule, Trainer

    from project.configs.config import Config
    from project.trainers.jax_trainer import JaxModule, JaxTrainer

logger = logging.getLogger(__name__)


# BUG: Always using the pydantic parser when instantiating things would be nice, but it currently
# causes issues related to pickling: https://github.com/mit-ll-responsible-ai/hydra-zen/issues/717
# def _use_pydantic[C: Callable](fn: C) -> C:
#     return functools.partial(hydra_zen.instantiate, _target_wrapper_=pydantic_parser)  # type: ignore
# instantiate = _use_pydantic(hydra_zen.instantiate)

instantiate = hydra_zen.instantiate


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


def instantiate_trainer(experiment_config: Config) -> Trainer | JaxTrainer:
    # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
    # fields have the right type.

    # instantiate all the callbacks
    callback_configs = experiment_config.trainer.pop("callbacks", {})
    callback_configs = {k: v for k, v in callback_configs.items() if v is not None}
    callbacks: dict[str, Callback] | None = hydra.utils.instantiate(
        callback_configs, _convert_="object"
    )
    # Create the loggers, if any.
    loggers: dict[str, Any] | None = hydra.utils.instantiate(
        experiment_config.trainer.pop("logger", {})
    )

    # Create the Trainer.

    # BUG: `hydra.utils.instantiate` doesn't work with override **kwargs when some of them are
    # dataclasses (e.g. a callback).
    # trainer = hydra.utils.instantiate(
    #     config,
    #     callbacks=list(callbacks.values()) if callbacks else None,
    #     logger=list(loggers.values()) if loggers else None,
    # )
    assert isinstance(experiment_config.trainer, dict)
    config = copy.deepcopy(experiment_config.trainer)
    target = hydra.utils.get_object(config.pop("_target_"))
    _callbacks = list(callbacks.values()) if callbacks else None
    _loggers = list(loggers.values()) if loggers else None

    trainer = target(**config, callbacks=_callbacks, logger=_loggers)
    return trainer


def instantiate_datamodule(
    datamodule_config: Builds[type[LightningDataModule]] | LightningDataModule | None,
) -> LightningDataModule | None:
    """Instantiate the datamodule from the configuration dict.

    Any interpolations in the config will have already been resolved by the time we get here.
    """
    if not datamodule_config:
        return None
    import lightning

    if isinstance(datamodule_config, lightning.LightningDataModule):
        logger.info(
            f"Datamodule was already instantiated (probably to interpolate a field value). "
            f"{datamodule_config=}"
        )
        datamodule = datamodule_config
    else:
        logger.debug(f"Instantiating datamodule from config: {datamodule_config}")
        datamodule = instantiate(datamodule_config)

    return datamodule


def instantiate_algorithm(
    algorithm_config: Config, datamodule: LightningDataModule | None
) -> LightningModule | JaxModule:
    """Function used to instantiate the algorithm.

    It is suggested that your algorithm (LightningModule) take in the `datamodule` and `network`
    as arguments, to make it easier to swap out different networks and datamodules during
    experiments.

    The instantiated datamodule and network will be passed to the algorithm's constructor.
    """
    # TODO: The algorithm is now always instantiated on the CPU, whereas it used to be instantiated
    # directly on the default device (GPU).
    # Create the algorithm
    algo_config = algorithm_config
    import lightning

    if isinstance(algo_config, lightning.LightningModule):
        logger.info(
            f"Algorithm was already instantiated (probably to interpolate a field value)."
            f"{algo_config=}"
        )
        return algo_config

    if datamodule:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config, datamodule=datamodule)
    else:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config)

    if isinstance(algo_or_algo_partial, functools.partial):
        if datamodule:
            algorithm = algo_or_algo_partial(datamodule=datamodule)
        else:
            algorithm = algo_or_algo_partial()
    else:
        # logger.warning(
        #     f"Your algorithm config {algo_config} doesn't have '_partial_: true' set, which is "
        #     f"not recommended (since we can't pass the datamodule to the constructor)."
        # )
        algorithm = algo_or_algo_partial
    from project.trainers.jax_trainer import JaxModule

    if not isinstance(algorithm, lightning.LightningModule | JaxModule):
        logger.warning(
            UserWarning(
                f"Your algorithm ({algorithm}) is not a LightningModule. Beware that this isn't "
                f"explicitly supported at the moment."
            )
        )

    return algorithm

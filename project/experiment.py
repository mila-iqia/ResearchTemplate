from __future__ import annotations

import functools
import logging
import typing
import warnings
from typing import Any

import hydra
import lightning
import rich
from hydra_zen.typing import Builds
from omegaconf import DictConfig

from project.configs.config import Config

if typing.TYPE_CHECKING:
    import lightning

    from project.trainers.jax_trainer import JaxTrainer


logger = logging.getLogger(__name__)


@functools.singledispatch
def train(
    algorithm,
    /,
    **kwargs,
) -> tuple[Any, Any]:
    raise NotImplementedError(
        f"There is no registered handler for training algorithm {algorithm} of type "
        f"{type(algorithm)}! (kwargs: {kwargs})."
        f"Registered handlers: "
        + "\n\t".join([f"- {k}: {v.__name__}" for k, v in train.registry.items()])
    )


@functools.singledispatch
def evaluate(algorithm: Any, /, **kwargs) -> tuple[str, float | None, dict]:
    """Evaluates the algorithm.

    Returns the name of the 'error' metric for this run, its value, and a dict of metrics.
    """
    raise NotImplementedError(
        f"There is no registered handler for evaluating algorithm {algorithm} of type "
        f"{type(algorithm)}! (kwargs: {kwargs})"
    )


def instantiate_trainer(trainer_config: dict | DictConfig) -> lightning.Trainer | JaxTrainer:
    # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
    # fields have the right type.
    # Create the Trainer
    trainer_config = trainer_config.copy()  # Avoid mutating the config.
    callbacks: list | None = instantiate_values(trainer_config.pop("callbacks", None))
    logger: list | None = instantiate_values(trainer_config.pop("logger", None))
    trainer = hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=logger)
    return trainer


def instantiate_values(config_dict: DictConfig | None) -> list[Any] | None:
    """Returns the list of objects at the values in this dict of configs.

    This is used for the config of the `trainer/logger` and `trainer/callbacks` fields, where
    we can combine multiple config groups by adding entries in a dict.

    For example, using `trainer/logger=wandb` and `trainer/logger=tensorboard` would result in a
    dict with `wandb` and `tensorboard` as keys, and the corresponding config groups as values.

    This would then return a list with the instantiated WandbLogger and TensorBoardLogger objects.
    """
    if not config_dict:
        return None
    objects_dict = hydra.utils.instantiate(config_dict, _recursive_=True)
    if objects_dict is None:
        return None

    assert isinstance(objects_dict, dict | DictConfig)
    return [v for v in objects_dict.values() if v is not None]


MetricName = str

import lightning  # noqa


@evaluate.register(lightning.LightningModule)
def evaluate_lightningmodule(
    algorithm: lightning.LightningModule,
    /,
    *,
    trainer: lightning.Trainer,
    datamodule: lightning.LightningDataModule | None = None,
    config: Config,
    train_results: Any = None,
) -> tuple[MetricName, float | None, dict]:
    """Evaluates the algorithm and returns the metrics.

    By default, if validation is to be performed, returns the validation error. Returns the
    training error when `trainer.overfit_batches != 0` (e.g. when debugging or testing). Otherwise,
    if `trainer.limit_val_batches == 0`, returns the test error.
    """
    datamodule = datamodule or getattr(algorithm, "datamodule", None)

    # exp.trainer.logger.log_hyperparams()
    # When overfitting on a single batch or only training, we return the train error.
    if (trainer.limit_val_batches == trainer.limit_test_batches == 0) or (
        trainer.overfit_batches == 1  # type: ignore
    ):
        # We want to report the training error.
        results_type = "train"
        results = [
            {
                **trainer.logged_metrics,
                **trainer.callback_metrics,
                **trainer.progress_bar_metrics,
            }
        ]
    elif trainer.limit_val_batches != 0:
        results_type = "val"
        results = trainer.validate(model=algorithm, datamodule=datamodule)
    else:
        warnings.warn(RuntimeWarning("About to use the test set for evaluation!"))
        results_type = "test"
        results = trainer.test(model=algorithm, datamodule=datamodule)

    if results is None:
        rich.print("RUN FAILED!")
        return "fail", None, {}

    metrics = dict(results[0])
    for key, value in metrics.items():
        rich.print(f"{results_type} {key}: ", value)

    if (accuracy := metrics.get(f"{results_type}/accuracy")) is not None:
        # NOTE: This is the value that is used for HParam sweeps.
        metric_name = "1-accuracy"
        error = 1 - accuracy

    elif (loss := metrics.get(f"{results_type}/loss")) is not None:
        logger.info("Assuming that the objective to minimize is the loss metric.")
        # If 'accuracy' isn't in the results, assume that the loss is the metric to use.
        metric_name = "loss"
        error = loss
    else:
        raise RuntimeError(
            f"Don't know which metric to use to calculate the 'error' of this run.\n"
            f"Here are the available metric names:\n"
            f"{list(metrics.keys())}"
        )

    return metric_name, error, metrics


def instantiate_datamodule(
    datamodule_config: Builds[type[lightning.LightningDataModule]]
    | lightning.LightningDataModule
    | None,
) -> lightning.LightningDataModule | None:
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
        return datamodule_config

    logger.debug(f"Instantiating datamodule from config: {datamodule_config}")
    return hydra.utils.instantiate(datamodule_config)


@train.register
def train_lightningmodule(
    algorithm: lightning.LightningModule,
    /,
    *,
    trainer: lightning.Trainer | None,
    datamodule: lightning.LightningDataModule | None = None,
    config: Config,
):
    # Create the Trainer from the config.
    if trainer is None:
        _trainer = instantiate_trainer(config.trainer)
        assert isinstance(_trainer, lightning.Trainer)
        trainer = _trainer

    # Train the model using the dataloaders of the datamodule:
    # The Algorithm gets to "wrap" the datamodule if it wants to. This could be useful for
    # example in RL, where we need to set the actor to use in the environment, as well as
    # potentially adding Wrappers on top of the environment, or having a replay buffer, etc.
    if datamodule is None:
        if hasattr(algorithm, "datamodule"):
            datamodule = getattr(algorithm, "datamodule")
        elif config.datamodule is not None:
            datamodule = instantiate_datamodule(config.datamodule)
    trainer.fit(algorithm, datamodule=datamodule, ckpt_path=config.ckpt_path)
    train_results = None  # todo: get the train results from the trainer.
    return algorithm, train_results

"""Training script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm), optionally datamodule and network;
3. Trains the model;
4. Optionally runs an evaluation loop.

"""

from __future__ import annotations

import dataclasses
import functools
import operator
import os
import warnings
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import hydra
import jax.random
import lightning
import omegaconf
import rich
from hydra_plugins.auto_schema import auto_schema_plugin
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from project.algorithms.jax_rl_example import EvalMetrics
from project.configs import add_configs_to_hydra_store
from project.configs.config import Config
from project.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    setup_logging,
)
from project.trainers.jax_trainer import JaxModule, JaxTrainer, Ts, _MetricsT
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.utils import print_config

logger = get_logger(__name__)

PROJECT_NAME = Path(__file__).parent.name
add_configs_to_hydra_store()
setup_logging(log_level="INFO", global_log_level="ERROR")


auto_schema_plugin.config = auto_schema_plugin.AutoSchemaPluginConfig(
    schemas_dir=REPO_ROOTDIR / ".schemas",
    regen_schemas=False,
    stop_on_error=False,
    quiet=True,
    verbose=False,
    add_headers=False,  # don't fallback to adding headers if we can't use vscode settings file.
)


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

    # Resolve all the interpolations in the configs.
    config: Config = resolve_dictconfig(dict_config)

    setup_logging(
        log_level=config.log_level,
        global_log_level="DEBUG" if config.debug else "INFO" if config.verbose else "WARNING",
    )

    # seed the random number generators, so the weights that are
    # constructed are deterministic and reproducible.
    lightning.seed_everything(seed=config.seed, workers=True)

    # Create the Trainer
    trainer_config = config.trainer.copy()  # Avoid mutating the config if possible.
    callbacks: list[Callback] | None = instantiate_values(trainer_config.pop("callbacks", None))
    logger: list[Logger] | None = instantiate_values(trainer_config.pop("logger", None))
    trainer: lightning.Trainer | JaxTrainer = hydra.utils.instantiate(
        trainer_config, callbacks=callbacks, logger=logger
    )

    # Create the datamodule (if present)
    datamodule: lightning.LightningDataModule | None = instantiate_datamodule(config.datamodule)

    # Create the "algorithm"
    algorithm: lightning.LightningModule | JaxModule = instantiate_algorithm(
        config.algorithm, datamodule=datamodule
    )

    import wandb

    if wandb.run:
        wandb.run.config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
        wandb.run.config.update(
            omegaconf.OmegaConf.to_container(dict_config, resolve=False, throw_on_missing=True)
        )
    # Train the algorithm.
    train_results = train(
        config=config, trainer=trainer, datamodule=datamodule, algorithm=algorithm
    )

    # Evaluate the algorithm.
    if isinstance(algorithm, JaxModule):
        assert isinstance(trainer, JaxTrainer)
        metric_name, error, _metrics = evaluate_jax_module(
            algorithm, trainer=trainer, train_results=train_results
        )
    else:
        assert isinstance(trainer, lightning.Trainer)
        metric_name, error, _metrics = evaluate_lightningmodule(
            algorithm, datamodule=datamodule, trainer=trainer
        )

    if wandb.run:
        wandb.finish()

    assert error is not None
    # Results are returned like this so that the Orion sweeper can parse the results correctly.
    return dict(name=metric_name, type="objective", value=error)


def train(
    config: Config,
    trainer: lightning.Trainer | JaxTrainer,
    datamodule: lightning.LightningDataModule | None,
    algorithm: lightning.LightningModule | JaxModule,
):
    if isinstance(trainer, lightning.Trainer):
        assert isinstance(algorithm, lightning.LightningModule)
        # Train the model using the dataloaders of the datamodule:
        # The Algorithm gets to "wrap" the datamodule if it wants to. This could be useful for
        # example in RL, where we need to set the actor to use in the environment, as well as
        # potentially adding Wrappers on top of the environment, or having a replay buffer, etc.
        datamodule = getattr(algorithm, "datamodule", datamodule)
        return trainer.fit(
            algorithm,
            datamodule=datamodule,
            ckpt_path=config.ckpt_path,
        )

    if datamodule is not None:
        raise NotImplementedError(
            "The JaxTrainer doesn't yet support using a datamodule. For now, you should "
            f"return a batch of data from the {JaxModule.get_batch.__name__} method in your "
            f"algorithm."
        )

    if not isinstance(algorithm, JaxModule):
        raise TypeError(
            f"The selected algorithm ({algorithm}) doesn't implement the required methods of "
            f"a {JaxModule.__name__}, so it can't be used with the `{JaxTrainer.__name__}`. "
            f"Try to subclass {JaxModule.__name__} and implement the missing methods."
        )
    rng = jax.random.key(config.seed)
    # TODO: Use ckpt_path argument to load the training state and resume the training run.
    assert config.ckpt_path is None
    return trainer.fit(algorithm, rng=rng)


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


def evaluate_lightningmodule(
    algorithm: lightning.LightningModule,
    trainer: lightning.Trainer,
    datamodule: lightning.LightningDataModule | None,
) -> tuple[MetricName, float | None, dict]:
    """Evaluates the algorithm and returns the metrics.

    By default, if validation is to be performed, returns the validation error. Returns the
    training error when `trainer.overfit_batches != 0` (e.g. when debugging or testing). Otherwise,
    if `trainer.limit_val_batches == 0`, returns the test error.
    """

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


def evaluate_jax_module(
    algorithm: JaxModule[Ts, Any, _MetricsT],
    trainer: JaxTrainer,
    train_results: tuple[Ts, _MetricsT] | None = None,
):
    # todo: there isn't yet a `validate` method on the jax trainer.
    assert isinstance(trainer, JaxTrainer)
    assert train_results is not None
    metrics = train_results[1]

    return get_error_from_metrics(metrics)


@functools.singledispatch
def get_error_from_metrics(metrics: _MetricsT) -> tuple[MetricName, float, dict]:
    """Returns the main metric name, its value, and the full metrics dictionary."""
    raise NotImplementedError(
        f"Don't know how to calculate the error to minimize from metrics {metrics} of type "
        f"{type(metrics)}! "
        f"You probably need to register a handler for it."
    )


@get_error_from_metrics.register(EvalMetrics)
def get_error_from_jax_rl_example_metrics(metrics: EvalMetrics):
    last_epoch_metrics = jax.tree.map(operator.itemgetter(-1), metrics)
    assert isinstance(last_epoch_metrics, EvalMetrics)
    # Average across eval seeds (we're doing evaluation in multiple environments in parallel with
    # vmap).
    last_epoch_average_cumulative_reward = last_epoch_metrics.cumulative_reward.mean().item()
    return (
        "-avg_cumulative_reward",
        -last_epoch_average_cumulative_reward,  # need to return an "error" to minimize for HPO.
        dataclasses.asdict(last_epoch_metrics),
    )


if __name__ == "__main__":
    main()

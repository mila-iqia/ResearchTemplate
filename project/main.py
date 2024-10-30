"""Training script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm), optionally datamodule and network;
3. Trains the model;
4. Optionally runs an evaluation loop.

"""

from __future__ import annotations

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
from lightning import Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from project.configs import add_configs_to_hydra_store
from project.configs.config import Config
from project.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    seed_rng,
    setup_logging,
)
from project.trainers.jax_trainer import JaxModule, JaxTrainer
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.utils import print_config

logger = get_logger(__name__)

PROJECT_NAME = Path(__file__).parent.name

add_configs_to_hydra_store()

auto_schema_plugin.config = auto_schema_plugin.AutoSchemaPluginConfig(
    schemas_dir=REPO_ROOTDIR / ".schemas",
    regen_schemas=False,
    stop_on_error=False,
    quiet=True,
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
    """
    print_config(dict_config, resolve=False)

    config: Config = resolve_dictconfig(dict_config)

    experiment_config = config
    setup_logging(experiment_config)
    seed_rng(experiment_config)

    trainer_config = config.trainer.copy()  # Avoid mutating the input config, if passed.
    callbacks: list[Callback] | None = instantiate_values(trainer_config.pop("callbacks", None))
    logger: list[Logger] | None = instantiate_values(trainer_config.pop("logger", None))

    trainer: lightning.Trainer | JaxTrainer = hydra.utils.instantiate(
        trainer_config, callbacks=callbacks, logger=logger
    )

    datamodule: lightning.LightningDataModule | None = instantiate_datamodule(
        experiment_config.datamodule
    )

    algorithm: lightning.LightningModule | JaxModule = instantiate_algorithm(
        experiment_config.algorithm, datamodule=datamodule
    )

    import wandb

    if wandb.run:
        wandb.run.config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
        wandb.run.config.update(
            omegaconf.OmegaConf.to_container(dict_config, resolve=False, throw_on_missing=True)
        )

    train(config=config, trainer=trainer, datamodule=datamodule, algorithm=algorithm)

    metric_name, error, _metrics = evaluation(
        trainer=trainer, datamodule=datamodule, algorithm=algorithm
    )

    if wandb.run:
        wandb.finish()

    assert error is not None
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
        trainer.fit(
            algorithm,
            datamodule=datamodule,
            ckpt_path=config.ckpt_path,
        )
        return

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
    trainer.fit(algorithm, rng=rng)


def instantiate_values(config_dict: DictConfig | None) -> list[Any] | None:
    """Returns the list of objects at the values in this dict of configs.

    This is used for the config of the `trainer/logger` and `trainer/callbacks` fields, where
    we can combine multiple config groups by adding entries in a dict.

    For example, using `trainer/logger=wandb` and `trainer/logger=tensorboard` would result in a
    dict with `wandb` and `tensorboard` as keys, and the corresponding config groups as values.

    This would then return a list with the instantiated WandbLogger and TensorBoardLogger objects.
    """
    if not config_dict:
        return []
    objects_dict = hydra.utils.instantiate(config_dict, _recursive_=True)
    if objects_dict is None:
        return None
    assert isinstance(objects_dict, dict | DictConfig)
    return [v for v in objects_dict.values() if v is not None]


MetricName = str


def evaluation(
    trainer: JaxTrainer | lightning.Trainer,
    datamodule: lightning.LightningDataModule | None,
    algorithm,
) -> tuple[MetricName, float | None, dict]:
    """Evaluates the algorithm and returns the metrics.

    By default, if validation is to be performed, returns the validation error. Returns the
    training error when `trainer.overfit_batches != 0` (e.g. when debugging or testing). Otherwise,
    if `trainer.limit_val_batches == 0`, returns the test error.
    """
    # TODO Probably log the hydra config with something like this:
    # exp.trainer.logger.log_hyperparams()
    # When overfitting on a single batch or only training, we return the train error.
    if (trainer.limit_val_batches == trainer.limit_test_batches == 0) or (
        trainer.overfit_batches == 1  # type: ignore
    ):
        # We want to report the training error.
        metrics = {
            **trainer.logged_metrics,
            **trainer.callback_metrics,
            **trainer.progress_bar_metrics,
        }
        rich.print(metrics)
        if "train/accuracy" in metrics:
            train_acc: float = metrics["train/accuracy"]
            train_error = 1 - train_acc
            return "1-accuracy", train_error, metrics
        elif "train/avg_episode_reward" in metrics:
            average_episode_rewards: float = metrics["train/avg_episode_reward"]
            train_error = -average_episode_rewards
            return "-avg_episode_reward", train_error, metrics
        elif "train/loss" in metrics:
            return "loss", metrics["train/loss"], metrics
        else:
            raise RuntimeError(
                f"Don't know which metric to use to calculate the 'error' of this run.\n"
                f"Here are the available metric names:\n"
                f"{list(metrics.keys())}"
            )
    assert isinstance(datamodule, LightningDataModule)

    if trainer.limit_val_batches != 0:
        results = trainer.validate(model=algorithm, datamodule=datamodule)
        results_type = "val"
    else:
        warnings.warn(RuntimeWarning("About to use the test set for evaluation!"))
        results = trainer.test(model=algorithm, datamodule=datamodule)
        results_type = "test"

    if results is None:
        rich.print("RUN FAILED!")
        return "fail", None, {}

    returned_results_dict = dict(results[0])
    results_dict = dict(results[0]).copy()

    loss = results_dict.pop(f"{results_type}/loss")

    if f"{results_type}/accuracy" in results_dict:
        accuracy: float = results_dict[f"{results_type}/accuracy"]
        rich.print(f"{results_type} accuracy: {accuracy:.1%}")

        if top5_accuracy := results_dict.get(f"{results_type}/top5_accuracy") is not None:
            rich.print(f"{results_type} top5 accuracy: {top5_accuracy:.1%}")
        # NOTE: This is the value that is used for HParam sweeps.
        error = 1 - accuracy
        metric_name = "1-accuracy"
    else:
        logger.warning("Assuming that the objective to minimize is the loss metric.")
        # If 'accuracy' isn't in the results, assume that the loss is the metric to use.
        metric_name = "loss"
        error = loss

    for key, value in results_dict.items():
        rich.print(f"{results_type} {key}: ", value)

    return metric_name, error, returned_results_dict


if __name__ == "__main__":
    main()

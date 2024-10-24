"""Main entry-point."""

from __future__ import annotations

import os
import warnings
from logging import getLogger as get_logger
from pathlib import Path

import hydra
import lightning
import omegaconf
import rich
from lightning import LightningDataModule
from omegaconf import DictConfig

from project.configs import add_configs_to_hydra_store
from project.configs.config import Config
from project.trainers.jax_trainer import JaxTrainer
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.typing_utils.protocols import DataModule
from project.utils.utils import print_config

logger = get_logger(__name__)

PROJECT_NAME = Path(__file__).parent.name

add_configs_to_hydra_store()


@hydra.main(
    config_path=f"pkg://{PROJECT_NAME}.configs",
    config_name="config",
    version_base="1.2",
)
def main(dict_config: DictConfig) -> dict:
    """Main entry point for training a model."""
    print_config(dict_config, resolve=False)

    from project.utils.auto_schema import add_schemas_to_all_hydra_configs

    # Note: running this should take ~5 seconds the first time and <1s after that.
    try:
        add_schemas_to_all_hydra_configs(
            config_files=None,
            repo_root=REPO_ROOTDIR,
            configs_dir=REPO_ROOTDIR / PROJECT_NAME / "configs",
            regen_schemas=False,
            stop_on_error=False,
            quiet=True,
            add_headers=False,  # don't add headers if we can't add an entry in vscode settings.
        )
    except Exception:
        logger.error("Unable to add schemas to all hydra configs.")

    config: Config = resolve_dictconfig(dict_config)

    from project.experiment import (
        instantiate_algorithm,
        instantiate_datamodule,
        instantiate_trainer,
        seed_rng,
        setup_logging,
    )

    experiment_config = config
    setup_logging(experiment_config)
    seed_rng(experiment_config)

    trainer = instantiate_trainer(experiment_config)

    datamodule = instantiate_datamodule(experiment_config.datamodule)

    algorithm = instantiate_algorithm(experiment_config.algorithm, datamodule=datamodule)
    # experiment: Experiment = setup_experiment(config)

    import wandb

    if wandb.run:
        wandb.run.config.update({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
        wandb.run.config.update(
            omegaconf.OmegaConf.to_container(dict_config, resolve=False, throw_on_missing=True)
        )

    # Train the model using the dataloaders of the datamodule:
    # The Algorithm gets to "wrap" the datamodule if it wants. This might be useful in the
    # case of RL, where we need to set the actor to use in the environment, as well as
    # potentially adding Wrappers on top of the environment, or having a replay buffer, etc.
    # TODO: Add ckpt_path argument to resume a training run.
    datamodule = getattr(algorithm, "datamodule", datamodule)

    if datamodule is None:
        from project.trainers.jax_trainer import JaxTrainer

        if isinstance(trainer, JaxTrainer):
            import jax.random

            trainer.fit(algorithm, rng=jax.random.key(0))  # type: ignore
        else:
            trainer.fit(algorithm)

    else:
        assert isinstance(datamodule, LightningDataModule)
        trainer.fit(
            algorithm,
            datamodule=datamodule,
        )

    metric_name, error, _metrics = evaluation(
        trainer=trainer, datamodule=datamodule, algorithm=algorithm
    )

    if wandb.run:
        wandb.finish()

    assert error is not None
    return dict(name=metric_name, type="objective", value=error)
    # return {metric_name: objective}


def evaluation(
    trainer: JaxTrainer | lightning.Trainer, datamodule: DataModule, algorithm
) -> tuple[str, float | None, dict]:
    """Return the classification error.

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

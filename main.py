from __future__ import annotations

import os
import warnings
from logging import getLogger as get_logger

import hydra
from hydra_zen import instantiate
from lightning import LightningDataModule
import rich
import wandb
from omegaconf import DictConfig, OmegaConf
from project.algorithms.rl_example.rl_datamodule import RlDataModule

from project.configs.config import Config
from project.experiment import Experiment, setup_experiment
from project.utils.utils import print_config

if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
    # NOTE: Perhaps unsetting it would also work, but this works atm.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = get_logger(__name__)


@hydra.main(
    config_path="project/configs",
    config_name="config",
    version_base="1.2",
)
def main(dict_config: Config) -> float:
    # Convert the "raw" DictConfig (which uses the `Config` class to define it's structure)
    # into an actual `Config` object:
    # config = OmegaConf.to_object(dict_config)
    # assert isinstance(config, Config)

    experiment: Experiment = setup_experiment(dict_config)

    objective, _metrics = run(experiment)
    assert objective is not None
    return objective


def run(experiment: Experiment) -> tuple[float | None, dict]:
    # Train the model using the dataloaders of the datamodule:
    # TODO: Should we use `algorithm.datamodule` instead? That way the Algorithm gets to "wrap"
    # the datamodule however it wants? This might be useful in the case of RL, the need to pass an
    # actor, as well as potentially adding Wrappers on top of the environment, etc. introduces some
    # coupling between the Algorithm and the DataModule.
    # experiment.trainer.fit(experiment.algorithm, datamodule=experiment.datamodule)

    # assert experiment.algorithm.datamodule is experiment.datamodule

    assert isinstance(experiment.algorithm.datamodule, LightningDataModule)
    experiment.trainer.fit(experiment.algorithm, datamodule=experiment.algorithm.datamodule)

    error, metrics = evaluation(experiment)
    if wandb.run:
        wandb.finish()
    return error, metrics


def evaluation(experiment: Experiment) -> tuple[float | None, dict]:
    """Return the classification error.

    By default, if validation is to be performed, returns the validation error. Returns the
    training error when `trainer.overfit_batches != 0` (e.g. when debugging or testing). Otherwise,
    if `trainer.limit_val_batches == 0`, returns the test error.
    """
    # TODO Probably log the hydra config with something like this:
    # exp.trainer.logger.log_hyperparams()
    # When overfitting on a single batch or only training, we return the train error.
    if (experiment.trainer.limit_val_batches == experiment.trainer.limit_test_batches == 0) or (
        experiment.trainer.overfit_batches == 1  # type: ignore
    ):
        # We want to report the training error.
        metrics = {
            **experiment.trainer.logged_metrics,
            **experiment.trainer.callback_metrics,
            **experiment.trainer.progress_bar_metrics,
        }
        rich.print(metrics)
        if "train/accuracy" in metrics:
            train_acc: float = metrics["train/accuracy"]
            train_error = 1 - train_acc
            return train_error, metrics
        elif "train/avg_episode_reward" in metrics:
            average_episode_rewards: float = metrics["train/avg_episode_reward"]
            train_error = -average_episode_rewards
            return train_error, metrics
        elif "train/loss" in metrics:
            return metrics["train/loss"], metrics
        else:
            raise RuntimeError(
                f"Don't know which metric to use to calculate the 'error' of this run.\n"
                f"Here are the available metric names:\n"
                f"{list(metrics.keys())}"
            )
    assert isinstance(experiment.datamodule, LightningDataModule)

    if experiment.trainer.limit_val_batches != 0:
        results = experiment.trainer.validate(
            model=experiment.algorithm, datamodule=experiment.datamodule
        )
        results_type = "val"
    else:
        warnings.warn(RuntimeWarning("About to use the test set for evaluation!"))
        results = experiment.trainer.test(
            model=experiment.algorithm, datamodule=experiment.datamodule
        )
        results_type = "test"

    if results is None:
        rich.print("RUN FAILED!")
        return None, {}

    returned_results_dict = results[0]
    results_dict = results[0].copy()
    # rich.print(results_dict)

    loss = results_dict.pop(f"{results_type}/loss")
    # rich.print(f"{results_type} loss: {loss:.2f}")
    if isinstance(experiment.datamodule, RlDataModule):
        # NOTE: Reinforce is a bit of an annoying algorithm, because it can be used in RL and SL..
        average_episode_rewards = results_dict.pop(f"{results_type}/avg_episode_reward")
        average_episode_returns = results_dict.pop(f"{results_type}/avg_episode_return")
        average_episode_length = results_dict.pop(f"{results_type}/avg_episode_length")
        rich.print(f"{results_type} Average episode rewards: {average_episode_rewards:.2f}")
        rich.print(f"{results_type} Average episode returns: {average_episode_returns:.2f}")
        rich.print(f"{results_type} Average episode length: {average_episode_length:.1}")
        # NOTE: This is the "lower is better" value that is used for HParam sweeps. Does it make
        # sense to use the (val/test) loss here? Or should we use -1 * rewards/returns?
        error = loss
    else:
        accuracy: float = results_dict.pop(f"{results_type}/accuracy")
        top5_accuracy: float = results_dict.pop(f"{results_type}/top5_accuracy")
        rich.print(f"{results_type} top1 accuracy: {accuracy:.1%}")
        rich.print(f"{results_type} top5 accuracy: {top5_accuracy:.1%}")
        # NOTE: This is the value that is used for HParam sweeps.
        error = 1 - accuracy

    for key, value in results_dict.items():
        rich.print(f"{results_type} {key}: ", value)

    return error, returned_results_dict


if __name__ == "__main__":
    main()

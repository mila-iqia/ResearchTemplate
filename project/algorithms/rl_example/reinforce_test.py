from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from hydra_zen import instantiate

from project.algorithms.algorithm_test import (
    AlgorithmTests,
    get_all_datamodule_names,
    get_all_network_names,
    get_experiment_config,
)
from project.algorithms.rl_example.reinforce import ExampleRLAlgorithm
from project.algorithms.rl_example.rl_datamodule import RlDataModule
from project.configs.config import Config
from project.experiment import setup_experiment
from project.main import run


class TestReinforce(AlgorithmTests[ExampleRLAlgorithm]):
    algorithm_type: type[ExampleRLAlgorithm] = ExampleRLAlgorithm
    algorithm_name: ClassVar[str] = "rl_example"

    unsupported_datamodule_names: ClassVar[list[str]] = list(
        set(get_all_datamodule_names()) - {"rl", "cartpole"}
    )
    unsupported_network_names: ClassVar[list[str]] = list(
        set(get_all_network_names()) - {"fcnet"}
    )

    @pytest.fixture(scope="class")
    def datamodule(self, hydra_options: Config) -> RlDataModule:
        datamodule = instantiate(hydra_options.datamodule)
        assert isinstance(datamodule, RlDataModule)
        # assert isinstance(datamodule, DataModule)
        return datamodule

    @pytest.fixture
    def algorithm(
        self, algorithm_kwargs: dict, datamodule: RlDataModule
    ) -> ExampleRLAlgorithm:
        algo = self.algorithm_cls(**algorithm_kwargs)
        assert algo.datamodule is datamodule
        return algo

    @pytest.mark.parametrize(
        (
            "rl_datamodule_name",
            "initial_state_seed",
            "num_training_iterations",
            "expected_reward",
        ),
        [
            (
                "pendulum",
                123,
                500,  # This is quite high, makes the test take very long to run.
                # TODO: Tune Reinforce on that env and create an override config for that
                # combination of network / datamodule / algorithm, and then increase this value.
                # Reinforce doesn't seem to be learning quick enough on Pendulum-v1 at the moment.
                -500,
            ),
            (
                "cartpole",
                123,
                100,  # TODO: This is a bit high, would be nice to reduce it.
                100,
            ),
        ],
    )
    def test_overfit_single_initial_state(
        self,
        network_name: str,
        tmp_path: Path,
        rl_datamodule_name: str,
        initial_state_seed: int,
        num_training_iterations: int,
        expected_reward: float,
    ) -> None:
        """Test that the algorithm can "overfit" to a given initial environment state.

        This is the equivalent to the "overfit one batch" test for SL algorithms, but in this case
        we set the initial state of the environment to always the same value, and repeatedly train
        the RL algorithm to achieve the highest reward possible from that state.

        NOTE: Currently, we have that at the start of every new epoch, the RlDataset resets the
        env with the seed that is passed to its constructor. The RlDataModule sets that seed with
        its `train_seed` argument. Therefore, if we run many epochs, but with only one episode
        per epoch, we should be able to "overfit" to that particular initial state of the
        environment.

        This is a very useful test, since if this doesn't pass, any training run using this
        algorithm is most likely useless.
        """
        algorithm_name = self.algorithm_name or self.algorithm_cls.__name__.lower()
        config = get_experiment_config(
            command_line_overrides=[
                f"algorithm={algorithm_name}",
                f"network={network_name}",
                f"datamodule={rl_datamodule_name}",
                "datamodule.episodes_per_epoch=1",
                "datamodule.batch_size=1",
                f"+datamodule.train_seed={initial_state_seed}",
                # f"+datamodule.valid_seed={initial_state_seed}",
                #
                f"trainer.max_epochs={num_training_iterations}",
                "+trainer.limit_val_batches=0",
                "+trainer.limit_test_batches=0",
                "trainer/callbacks=no_checkpoints",
                "+trainer.enable_checkpointing=false",
                f"++trainer.default_root_dir={tmp_path}",
                "~trainer/logger",
                "seed=12345",
            ]
        )
        assert config.trainer["max_epochs"] == num_training_iterations

        experiment = setup_experiment(config)
        _, metrics = run(experiment)
        # Get the validation metric if present, else the training metric.
        average_episode_reward = metrics.get(
            "val/avg_episode_reward", metrics.get("train/avg_episode_reward")
        )
        assert average_episode_reward is not None
        assert average_episode_reward > expected_reward

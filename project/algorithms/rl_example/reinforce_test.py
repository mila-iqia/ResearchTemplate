from __future__ import annotations

import logging
import random
import time
from collections import Counter
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar

import gymnasium
import gymnasium.spaces
import matplotlib
import matplotlib.axes
import numpy as np
import pytest
import rich.logging
import torch
import tqdm
from hydra import compose, initialize_config_module
from lightning import Trainer
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from project.algorithms.rl_example.reinforce import (
    MeasureThroughputCallback,
    Reinforce,
    collect_episodes,
)
from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.configs.config import Config
from project.datamodules.rl.datamodule import RlDataModule
from project.datamodules.rl.envs import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.types import (
    EpisodeBatch,
    random_actor,
)
from project.experiment import setup_experiment
from project.main import run
from project.networks.fcnet import FcNet
from project.utils.device import default_device
from project.utils.testutils import run_for_all_configs_of_type

logger = get_logger(__name__)


def get_experiment_config(command_line_overrides: list[str]) -> Config:
    print(f"overrides: {' '.join(command_line_overrides)}")
    with initialize_config_module("beyond_backprop.configs"):
        config = compose(
            config_name="config",
            overrides=command_line_overrides,
        )

    config = OmegaConf.to_object(config)
    assert isinstance(config, Config)
    return config


@run_for_all_configs_of_type("datamodule", RlDataModule)
@run_for_all_configs_of_type("network", FcNet)
class TestReinforce(LearningAlgorithmTests[Reinforce]):
    metric_name: ClassVar[str] = "train/avg_episode_return"
    lower_is_better: ClassVar[bool] = False

    @pytest.mark.skip(
        reason=(
            "TODO: Double-check this, but I think thhe 'test overfit one batch' test already "
            "does pretty much the same thing as this here."
        )
    )
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
        _, _, metrics = run(experiment)
        # Get the validation metric if present, else the training metric.
        average_episode_reward = metrics.get(
            "val/avg_episode_reward", metrics.get("train/avg_episode_reward")
        )
        assert average_episode_reward is not None
        assert average_episode_reward > expected_reward


@pytest.mark.timeout(300)
@pytest.mark.parametrize("no_training", [True, False])
def test_benchmark_training_step(
    no_training: bool, device: torch.device, benchmark: BenchmarkFixture
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s ",
        handlers=[rich.logging.RichHandler()],
    )

    env_id = "CartPole-v1"
    device = default_device()
    # device = torch.device("cpu")
    # max_num_updates = 3
    num_envs = 1
    min_num_transitions_per_update = 10_000

    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.set_deterministic_debug_mode("warn")

    if num_envs is not None:
        assert num_envs >= 1
        env = make_torch_vectorenv(env_id, num_envs=num_envs, seed=seed, device=device)
        env.single_observation_space.seed(seed)
        env.single_action_space.seed(seed)
    else:
        env = make_torch_env(env_id, seed=seed, device=device)
        num_envs = 1
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    with device:
        single_action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
        network = FcNet(
            # todo: change to `input_dims` and pass flatdim(observation_space) instead.
            output_dims=gymnasium.spaces.flatdim(single_action_space),
        )
        algorithm = Reinforce(datamodule=None, network=network)
        optimizer = algorithm.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Optimizer)

    def _training_step(episodes: EpisodeBatch):
        optimizer.zero_grad()
        step_output = algorithm.training_step(episodes)
        assert "loss" in step_output
        loss = step_output["loss"]
        assert isinstance(loss, Tensor) and loss.requires_grad
        loss.backward()
        optimizer.step()
        return loss

    episodes = collect_episodes(
        env,
        actor=random_actor if no_training else algorithm,
        min_num_transitions=min_num_transitions_per_update,
    )
    benchmark(_training_step, episodes)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("no_training", [True, False])
def test_steps_per_second(no_training: bool, device: torch.device):
    import logging

    import rich.logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s ",
        handlers=[rich.logging.RichHandler()],
    )

    env_id = "CartPole-v1"
    device = default_device()
    # device = torch.device("cpu")
    max_num_updates = 3
    num_envs = 1
    min_num_transitions_per_update = 10_000

    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.set_deterministic_debug_mode("warn")

    if num_envs is not None:
        assert num_envs >= 1
        env = make_torch_vectorenv(env_id, num_envs=num_envs, seed=seed, device=device)
        env.single_observation_space.seed(seed)
        env.single_action_space.seed(seed)
    else:
        env = make_torch_env(env_id, seed=seed, device=device)
        num_envs = 1
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    with device:
        single_action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
        network = FcNet(
            # todo: change to `input_dims` and pass flatdim(observation_space) instead.
            output_dims=gymnasium.spaces.flatdim(single_action_space),
        )
        algorithm = Reinforce(datamodule=None, network=network)
        import tqdm

        optimizer = algorithm.configure_optimizers()

    pbar = tqdm.tqdm(range(max_num_updates))
    start = time.perf_counter()
    total_transitions = 0
    total_episodes = 0
    sps = 0
    updates_per_second = 0
    episodes_per_second = 0

    for update in pbar:
        episodes = collect_episodes(
            env,
            actor=random_actor if no_training else algorithm,
            min_num_transitions=min_num_transitions_per_update,
        )
        if no_training:
            continue
        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizer.zero_grad()
        step_output = algorithm.training_step(episodes)
        assert "loss" in step_output
        loss = step_output["loss"]
        assert isinstance(loss, Tensor)
        assert "log" in step_output
        logs = step_output["log"]
        assert logs, step_output
        loss.backward()
        optimizer.step()
        num_episodes = episodes.batch_size
        num_transitions = sum(episodes.episode_lengths)
        total_transitions += num_transitions
        total_episodes += num_episodes
        sps = total_transitions / (time.perf_counter() - start)
        updates_per_second = (update + 1) / (time.perf_counter() - start)
        episodes_per_second = total_episodes / (time.perf_counter() - start)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.2f}",
                "sps": sps,
                "ups": updates_per_second,
                "eps": episodes_per_second,
                **{k: v.item() if isinstance(v, Tensor) else v for k, v in logs.items()},
            }
        )

        if logs["avg_episode_length"] > 200:
            logger.info(
                f"Reached the threshold of an episode length of 200 after {update+1} updates "
                f"({logs['avg_episode_length']})."
            )
            break

    print(
        f"Num envs: {num_envs}, transitions per second: {sps}, updates per second: {updates_per_second}"
    )


@pytest.mark.timeout(300)
def test_compare_episode_lengths():
    import logging

    import rich.logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s ",
        handlers=[rich.logging.RichHandler()],
    )

    env_id = "CartPole-v1"
    device = default_device()
    num_envs = 16
    num_updates = 10

    fig, ax = plt.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    for num_envs in [1, 16, 64]:
        episode_lengths = get_episode_lengths(
            env_id, device=device, num_updates=num_updates, num_envs=num_envs
        )
        x = np.array(sorted(episode_lengths.keys()))
        y = np.array([episode_lengths[k] for k in x])
        label = "single_env" if num_envs == 1 else f"{num_envs} envs"
        ax.bar(x, y, label=label)
    fig.legend()
    fig.savefig("episode_lengths_comparison.png")
    fig.show()


def get_episode_lengths(
    env_id: str,
    num_envs: int,
    device=default_device(),
    num_updates: int = 3,
    min_num_transitions_per_update=10_000,
    seed: int = 123,
) -> Counter[int]:
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.set_deterministic_debug_mode("warn")

    if num_envs is not None:
        assert num_envs >= 1
        env = make_torch_vectorenv(env_id, num_envs=num_envs, seed=seed, device=device)
        env.single_observation_space.seed(seed)
        env.single_action_space.seed(seed)
    else:
        env = make_torch_env(env_id, seed=seed, device=device)
        num_envs = 1
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    with device:
        single_action_space = getattr(env.unwrapped, "single_action_space", env.action_space)
        network = FcNet(
            # todo: change to `input_dims` and pass flatdim(observation_space) instead.
            output_dims=gymnasium.spaces.flatdim(single_action_space),
        )
        algorithm = Reinforce(datamodule=None, network=network)

        optimizer = algorithm.configure_optimizers()

    pbar = tqdm.tqdm(range(num_updates))
    start = time.perf_counter()
    total_transitions = 0
    total_episodes = 0
    sps = 0
    updates_per_second = 0
    episodes_per_second = 0
    episode_lengths: Counter[int] = Counter()

    for update in pbar:
        episodes = collect_episodes(
            env,
            actor=algorithm,
            min_num_transitions=min_num_transitions_per_update,
        )
        episode_lengths.update(episodes.episode_lengths)

        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizer.zero_grad()
        step_output = algorithm.training_step(episodes)
        assert "loss" in step_output
        loss = step_output["loss"]
        assert isinstance(loss, Tensor)
        assert "log" in step_output
        logs = step_output["log"]
        assert logs, step_output
        loss.backward()
        optimizer.step()
        num_episodes = episodes.batch_size
        num_transitions = sum(episodes.episode_lengths)
        total_transitions += num_transitions
        total_episodes += num_episodes
        sps = total_transitions / (time.perf_counter() - start)
        updates_per_second = (update + 1) / (time.perf_counter() - start)
        episodes_per_second = total_episodes / (time.perf_counter() - start)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.2f}",
                "sps": sps,
                "ups": updates_per_second,
                "eps": episodes_per_second,
                **{k: v.item() if isinstance(v, Tensor) else v for k, v in logs.items()},
            }
        )

    return episode_lengths


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["CartPole-v1"])
@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_updates", [3])
def test_train_with_trainer(
    env_id: str,
    num_envs: int,
    device: torch.device,
    seed: int,
    num_updates: int,
):
    import logging

    import rich.logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s ",
        handlers=[rich.logging.RichHandler()],
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.set_deterministic_debug_mode("warn")

    datamodule = RlDataModule(
        env=env_id,
        batch_size=(
            batch_size := 2
        ),  # todo: fix this so it's something more like `min_transitions_per_batch` or `episodes_per_batch`.
        # todo: double-check that `episodes_per_epoch` takes the "batch size" into account. Probably rename it to `batches_per_epoch` instead.
        episodes_per_epoch=batch_size * num_updates,
        num_parallel_envs=num_envs,
    )
    with device:
        single_action_space = getattr(
            datamodule.env.unwrapped, "single_action_space", datamodule.env.action_space
        )
        network = FcNet(
            # todo: change to `input_dims` and pass flatdim(observation_space) instead.
            output_dims=gymnasium.spaces.flatdim(single_action_space),
        )
        algorithm = Reinforce(datamodule=datamodule, network=network)

    trainer = Trainer(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        reload_dataloaders_every_n_epochs=1,
        callbacks=[MeasureThroughputCallback()],
        detect_anomaly=False,
    )
    # todo: fine for now, but perhaps the SL->RL wrapper for Reinforce will change that.
    assert algorithm.datamodule is datamodule
    trainer.fit(algorithm, datamodule=datamodule)

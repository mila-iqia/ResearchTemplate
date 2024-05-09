import pytest
import torch

from project.datamodules.rl.envs.env_tests import EnvTests

from .debug_env import DebugEnv, DebugVectorEnv


@pytest.mark.timeout(3)
def test_action_affects_state(seed: int, device: torch.device):
    target = 3
    initial_state = 0
    env = DebugEnv(
        min=-10,
        max=10,
        max_episode_length=5,
        target=target,
        initial_state=initial_state,
        seed=seed,
        device=device,
        randomize_initial_state=False,
        randomize_target=False,
        wrap_around_state=True,
    )

    assert env.reset(seed=seed) == (
        0,
        {"episode_length": 0, "target": target},
    )

    def _action(v: int):
        return torch.tensor(v, dtype=env.action_space.dtype, device=device)

    # Increment
    assert env.step(_action(1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 1, "target": target},
    )
    # Stay the same
    assert env.step(_action(0)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 2, "target": target},
    )
    # Increment
    assert env.step(_action(1)) == (
        2,
        -1,
        False,
        False,
        {"episode_length": 3, "target": target},
    )
    # Decrement
    assert env.step(_action(-1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 4, "target": target},
    )
    # Decrement
    assert env.step(_action(-1)) == (
        0,
        -3,
        False,
        True,  # truncated because we weren't at the target state at the end.
        {"episode_length": 5, "target": target},
    )

    assert env.reset(seed=seed) == (
        0,
        {"episode_length": 0, "target": target},
    )
    assert env.step(_action(1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 1, "target": target},
    )
    assert env.step(_action(1)) == (
        2,
        -1,
        False,
        False,
        {"episode_length": 2, "target": target},
    )
    assert env.step(_action(1)) == (
        3,
        0,
        False,
        False,
        {"episode_length": 3, "target": target},
    )
    assert env.step(_action(1)) == (
        4,
        -1,  # important that reward is -abs(target - state))
        False,
        False,
        {"episode_length": 4, "target": target},
    )
    assert env.step(_action(-1)) == (
        3,
        0,
        True,  # terminated=True because we're on the target at the last step.
        False,
        {"episode_length": 5, "target": target},
    )


@pytest.mark.timeout(3)
def test_action_affects_vectorenv_state(seed: int, device: torch.device):
    target = 3
    initial_state = 0
    env = DebugVectorEnv(
        num_envs=2,
        min=-10,
        max=10,
        max_episode_length=5,
        target=target,
        initial_state=initial_state,
        device=device,
        randomize_initial_state=False,
        randomize_target=False,
        wrap_around_state=True,
    )

    assert env.reset(seed=seed) == (
        0,
        {"episode_length": 0, "target": target},
    )

    def _action(v: int):
        return torch.tensor(v, dtype=env.action_space.dtype, device=device)

    # Increment
    assert env.step(_action(1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 1, "target": target},
    )
    # Stay the same
    assert env.step(_action(0)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 2, "target": target},
    )
    # Increment
    assert env.step(_action(1)) == (
        2,
        -1,
        False,
        False,
        {"episode_length": 3, "target": target},
    )
    # Decrement
    assert env.step(_action(-1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 4, "target": target},
    )
    # Decrement
    assert env.step(_action(-1)) == (
        0,
        -3,
        False,
        True,  # truncated because we weren't at the target state at the end.
        {"episode_length": 5, "target": target},
    )

    assert env.reset(seed=seed) == (
        0,
        {"episode_length": 0, "target": target},
    )
    assert env.step(_action(1)) == (
        1,
        -2,
        False,
        False,
        {"episode_length": 1, "target": target},
    )
    assert env.step(_action(1)) == (
        2,
        -1,
        False,
        False,
        {"episode_length": 2, "target": target},
    )
    assert env.step(_action(1)) == (
        3,
        0,
        False,
        False,
        {"episode_length": 3, "target": target},
    )
    assert env.step(_action(1)) == (
        4,
        -1,  # important that reward is -abs(target - state))
        False,
        False,
        {"episode_length": 4, "target": target},
    )
    assert env.step(_action(-1)) == (
        3,
        0,
        True,  # terminated=True because we're on the target at the last step.
        False,
        {"episode_length": 5, "target": target},
    )


@pytest.mark.parametrize("env_id", ["debug"], indirect=True)
class TestDebugEnv(EnvTests): ...

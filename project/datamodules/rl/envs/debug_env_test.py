import numpy as np
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
        device=device,
        randomize_initial_state=False,
        randomize_target=False,
        wrap_around_state=True,
        seed=seed,
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

    def _to_list(v):
        if isinstance(v, list):
            return list(map(_to_list, v))
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return v.cpu().tolist()
        assert isinstance(v, dict)
        return {k: _to_list(v_i) for k, v_i in v.items()}

    def _reset(seed: int):
        obs, info = env.reset(seed=seed)
        return (
            _to_list(obs),
            {k: _to_list(v) for k, v in info.items()},
        )

    def _action(v: list[int]):
        return torch.tensor(v, dtype=env.action_space.dtype, device=device)

    def _step(action: list[int]):
        obs, reward, terminated, truncated, info = env.step(_action(action))

        assert isinstance(reward, torch.Tensor)
        return (
            _to_list(obs),
            _to_list(reward),
            _to_list(terminated),
            _to_list(truncated),
            {k: _to_list(v) for k, v in info.items()},
        )

    assert _reset(seed=seed) == (
        [0, 0],
        {"episode_length": [0, 0], "target": [target, target]},
    )
    # Increment and decrement
    assert _step([1, -1]) == (
        [1, -1],
        [-2, -4],
        [False, False],
        [False, False],
        {"episode_length": [1, 1], "target": [target, target]},
    )
    # Stay the same and decrement
    assert _step([0, -1]) == (
        [1, -2],
        [-2, -5],
        [False, False],
        [False, False],
        {"episode_length": [2, 2], "target": [target, target]},
    )
    assert _step([1, 1]) == (
        [2, -1],
        [-1, -4],
        [False, False],
        [False, False],
        {"episode_length": [3, 3], "target": [target, target]},
    )
    assert _step([1, 1]) == (
        [3, 0],
        [0, -3],
        [False, False],
        [False, False],
        {"episode_length": [4, 4], "target": [target, target]},
    )
    # NOTE: This next step is slightly weird, just like in gymnasium.vector.SyncVectorEnv.
    assert _step([0, 1]) == (
        [0, 0],  # Observations after reset
        [0, -2],  # Rewards from the last step
        [True, False],  # first env is on target
        [False, True],  # second env is truncated
        {
            # important: Auto-resets sub-envs just like gymnasium.vector.SyncVectorEnv.
            "episode_length": [0, 0],
            "target": [target, target],
            # Mask indicating whether each index is set or not.
            "_final_observation": [True, True],
            "final_observation": [3, 1],  # list of observations from the last step.
            "_final_info": [True, True],
            "final_info": [
                {
                    "episode_length": 5,
                    "target": target,
                },
                {
                    "episode_length": 5,
                    "target": target,
                },
            ],
        },
    )
    assert _step([1, 1]) == (  # important: Auto-reset should happen in each sub-env!
        [1, 1],
        [-2, -2],
        [False, False],
        [False, False],
        {"episode_length": [1, 1], "target": [target, target]},
    )


@pytest.mark.parametrize("env_id", ["debug"], indirect=True)
class TestDebugEnv(EnvTests): ...

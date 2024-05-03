from typing import Any

import gym
import gym.vector
import pytest
import torch
from torch import Tensor

from project.datamodules.rl.gym_utils import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.rl_types import VectorEnv
from project.utils.types import NestedDict


@pytest.fixture(params=["Pendulum-v1"])
def env_id(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[123])
def seed(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture()
def env(env_id: str, seed: int, device: torch.device):
    return make_torch_env(env_id, device=device, seed=seed)


@pytest.fixture(params=[1, 11, 16])
def num_envs(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture()
def vectorenv(env_id: str, seed: int, num_envs: int, device: torch.device):
    return make_torch_vectorenv(env_id=env_id, num_envs=num_envs, seed=seed, device=device)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "halfcheetah"], indirect=True)
def test_jax_env(env: gym.Env[torch.Tensor, torch.Tensor], seed: int, device: torch.device):
    obs_from_reset, info_from_reset = env.reset(seed=seed)

    def _check_obs(obs: Any):
        assert isinstance(obs, Tensor) and obs.device == device
        assert obs in env.observation_space

    def _check_dict(d: NestedDict[str, Tensor | Any]):
        for k, value in d.items():
            if isinstance(value, dict):
                _check_dict(value)
            elif value is not None:
                assert isinstance(value, Tensor) and value.device == device, k

    _check_obs(obs_from_reset)
    _check_dict(info_from_reset)

    obs_from_space = env.observation_space.sample()
    _check_obs(obs_from_space)

    action = env.action_space.sample()
    assert isinstance(action, torch.Tensor) and action.device == device
    assert action in env.action_space

    obs_from_step, reward, done, _trunc, info_from_step = env.step(action)
    _check_obs(obs_from_step)
    assert (
        isinstance(reward, torch.Tensor)
        and reward.device == device
        and reward.dtype == torch.float32
    )

    assert isinstance(done, torch.Tensor) and done.device == device and done.dtype == torch.bool
    assert (
        isinstance(_trunc, torch.Tensor) and _trunc.device == device and _trunc.dtype == torch.bool
    )
    _check_dict(info_from_step)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "halfcheetah"], indirect=True)
def test_jax_vectorenv(
    vectorenv: VectorEnv[torch.Tensor, torch.Tensor],
    num_envs: int,
    device: torch.device,
    seed: int,
):
    assert vectorenv.num_envs == num_envs

    obs_batch_from_reset, info_batch_from_reset = vectorenv.reset(seed=seed)

    def _check_obs(obs: Any):
        assert isinstance(obs, Tensor) and obs.device == device and obs.shape[0] == num_envs
        assert obs in vectorenv.observation_space
        assert all(obs_i in vectorenv.single_observation_space for obs_i in obs)

    def _check_dict(d: NestedDict[str, Tensor | Any]):
        for k, value in d.items():
            if isinstance(value, dict):
                _check_dict(value)
            elif value is not None:
                assert (
                    isinstance(value, Tensor)
                    and value.device == device
                    and value.shape[0] == num_envs
                ), k

    _check_obs(obs_batch_from_reset)
    _check_dict(info_batch_from_reset)

    obs_from_space = vectorenv.observation_space.sample()
    _check_obs(obs_from_space)

    action = vectorenv.action_space.sample()
    assert isinstance(action, torch.Tensor) and action.device == device
    assert action in vectorenv.action_space

    obs_from_step, reward, done, _trunc, info_from_step = vectorenv.step(action)
    _check_obs(obs_from_step)
    assert (
        isinstance(reward, torch.Tensor)
        and reward.device == device
        and reward.dtype == torch.float32
        and reward.shape == (num_envs,)
    )

    assert (
        isinstance(done, torch.Tensor)
        and done.device == device
        and done.dtype == torch.bool
        and done.shape == (num_envs,)
    )
    assert (
        isinstance(_trunc, torch.Tensor)
        and _trunc.device == device
        and _trunc.dtype == torch.bool
        and _trunc.shape == (num_envs,)
    )
    _check_dict(info_from_step)

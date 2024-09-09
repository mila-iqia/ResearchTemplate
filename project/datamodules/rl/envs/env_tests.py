from typing import Any

import gymnasium
import jax
import numpy as np
import pytest
import torch
import torch.utils
import torch.utils.data
from pytest_benchmark.fixture import BenchmarkFixture
from tensor_regression import TensorRegressionFixture
from torch import Tensor
from torch_jax_interop.to_torch import jax_to_torch_tensor

from project.datamodules.rl.envs import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.types import (
    Episode,
    EpisodeBatch,
    VectorEnv,
)
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete
from project.utils.typing_utils import NestedDict
from project.utils.utils import get_shape_ish

pytest.register_assert_rewrite(__file__)


class EnvTests:
    """Tests for the RL environments whose observations / actions are on the GPU."""

    @pytest.fixture(scope="class")
    def env_id(self, request: pytest.FixtureRequest):
        env_id_str = getattr(request, "param", None)
        if not env_id_str:
            raise RuntimeError(
                "You are supposed to pass the env_id via an indirect parametrization!"
            )
        return env_id_str

    @pytest.fixture(scope="class", params=[123])
    def seed(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture(scope="function")
    def env(self, env_id: str, seed: int, device: torch.device):
        return make_torch_env(env_id, device=device, seed=seed)

    @pytest.fixture(scope="class", params=[1, 11, 16])
    def num_envs(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(scope="function")
    def vectorenv(self, env_id: str, seed: int, num_envs: int, device: torch.device):
        return make_torch_vectorenv(env_id=env_id, num_envs=num_envs, seed=seed, device=device)

    @pytest.mark.timeout(30)
    def test_env(
        self,
        env: gymnasium.Env[torch.Tensor, torch.Tensor],
        seed: int,
        device: torch.device,
        tensor_regression: TensorRegressionFixture,
    ):
        observation_from_reset, info_from_reset = env.reset(seed=seed)

        def _check_observation(obs: Any):
            assert isinstance(obs, Tensor) and obs.device == device
            # todo: fix issue with `inf` in the cartpole observations of Gymnax.
            assert obs in env.observation_space, (obs, type(obs), env.observation_space)

        def _check_dict(d: NestedDict[str, Tensor | Any]):
            for k, value in d.items():
                if isinstance(value, dict):
                    _check_dict(value)
                elif value is not None:
                    assert isinstance(value, Tensor) and value.device == device, k

        _check_observation(observation_from_reset)
        _check_dict(info_from_reset)

        observation_from_space = env.observation_space.sample()
        _check_observation(observation_from_space)

        action_from_space = env.action_space.sample()
        assert isinstance(action_from_space, torch.Tensor) and action_from_space.device == device
        assert action_from_space in env.action_space

        observation_from_step, reward, terminated, truncated, info_from_step = env.step(
            action_from_space
        )
        _check_observation(observation_from_step)

        def _check_tensor_or_jax_array(
            v: Any, device: torch.device, dtype: torch.dtype
        ) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | jax.Array), v
            if isinstance(v, jax.Array):
                v = jax_to_torch_tensor(v)
            assert v.device == device
            assert v.dtype == dtype
            return v

        reward = _check_tensor_or_jax_array(reward, device, torch.float32)

        def _check_bool_or_tensor_or_jax_array(v: Any, device: torch.device, dtype: torch.dtype):
            assert isinstance(v, torch.Tensor | jax.Array | bool), v
            if isinstance(v, jax.Array):
                v = jax_to_torch_tensor(v)
            if isinstance(v, torch.Tensor):
                assert v.device == device
                assert v.dtype == dtype
            return v

        terminated = _check_bool_or_tensor_or_jax_array(terminated, device, torch.bool)
        truncated = _check_bool_or_tensor_or_jax_array(truncated, device, torch.bool)

        _check_dict(info_from_step)

        tensor_regression.check(
            {
                "obs_from_reset": observation_from_reset,
                "info_from_reset": info_from_reset,
                "obs_from_space": observation_from_space,
                "action_from_space": action_from_space,
                "obs_from_step": observation_from_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info_from_step": info_from_step,
            }
        )

    def test_reset_speed(
        self,
        vectorenv: VectorEnv[torch.Tensor, torch.Tensor],
        num_envs: int,
        device: torch.device,
        seed: int,
        tensor_regression: TensorRegressionFixture,
        benchmark: BenchmarkFixture,
    ):
        _ = benchmark(vectorenv.reset, seed=seed)

    def test_step_speed(
        self,
        vectorenv: VectorEnv[torch.Tensor, torch.Tensor],
        seed: int,
        benchmark: BenchmarkFixture,
    ):
        _ = vectorenv.reset(seed=seed)

        def _step():
            _ = vectorenv.step(vectorenv.action_space.sample())

        benchmark(_step)

    def test_vectorenv(
        self,
        vectorenv: VectorEnv[torch.Tensor, torch.Tensor],
        num_envs: int,
        device: torch.device,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        assert vectorenv.num_envs == num_envs

        obs_batch_from_reset, info_batch_from_reset = vectorenv.reset(seed=seed)

        def _check_obs(obs: Any):
            assert isinstance(obs, Tensor) and obs.device == device and obs.shape[0] == num_envs
            assert obs.dtype == vectorenv.observation_space.dtype
            assert obs.shape == vectorenv.observation_space.shape
            assert obs in vectorenv.observation_space, (obs, vectorenv.observation_space)
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

        action_from_space = vectorenv.action_space.sample()
        assert isinstance(action_from_space, torch.Tensor) and action_from_space.device == device
        assert action_from_space in vectorenv.action_space

        obs_from_step, reward, terminated, truncated, info_from_step = vectorenv.step(
            action_from_space
        )
        _check_obs(obs_from_step)

        assert isinstance(reward, torch.Tensor | jax.Array), reward
        if isinstance(reward, jax.Array):
            reward = jax_to_torch_tensor(reward)
        assert reward.shape == (num_envs,)
        assert reward.device == device
        assert reward.dtype == torch.float32

        assert isinstance(terminated, torch.Tensor | jax.Array), terminated
        assert terminated.shape == (num_envs,)
        if isinstance(terminated, jax.Array):
            terminated = jax_to_torch_tensor(terminated)
        assert terminated.device == device
        assert terminated.dtype == torch.bool

        assert isinstance(truncated, torch.Tensor | jax.Array), truncated
        assert truncated.shape == (num_envs,)
        if isinstance(truncated, jax.Array):
            truncated = jax_to_torch_tensor(truncated)
        assert truncated.device == device
        assert truncated.dtype == torch.bool

        _check_dict(info_from_step)

        tensor_regression.check(
            {
                "obs_from_reset": obs_batch_from_reset,
                "info_from_reset": info_batch_from_reset,
                "obs_from_space": obs_from_space,
                "action_from_space": action_from_space,
                "obs_from_step": obs_from_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info_from_step": info_from_step,
            }
        )

    def test_vectorenv_info_on_episode_end(
        self, vectorenv: VectorEnv[torch.Tensor, torch.Tensor], seed: int
    ):
        env = vectorenv
        n = vectorenv.num_envs
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        obs, infos = env.reset()
        # print(obs, infos)
        episode_step = 0
        max_episode_steps = 2000
        while "final_observation" not in infos:
            obs, reward, terminated, truncated, infos = env.step(env.action_space.sample())
            episode_step += 1
            assert episode_step < max_episode_steps

            # print(obs, reward, terminated, truncated, infos)
            # Ah HA! Every step gives these extra values in the info dict, not just the last one!
            keys_with_mask = sorted(k for k in infos.keys() if f"_{k}" in infos)
            mask_keys = [f"_{k}" for k in keys_with_mask]

            # IF some info is spawned inside an individual environment but maybe not in all envs,
            # then we should have a mask (following the SyncVectorEnv way of doing things).
            # If we don't find a mask for a particular entry, then we can assume it's already
            # vectorized and is present in all envs.
            regular_keys = set(infos.keys()) - set(mask_keys) - set(keys_with_mask)
            for key in regular_keys:
                _value = infos[key]
                # value should be either a tensor or a nested dict of tensors (or maybe None?)
                # assert isinstance(value, torch.Tensor)

            for mask_key, key in zip(mask_keys, keys_with_mask):
                assert mask_key == f"_{key}"
                mask = infos[mask_key]
                info = infos[key]
                assert (
                    isinstance(mask, jax.Array)
                    and str(mask.devices().pop()) == "cuda:0"
                    and mask.dtype == jax.numpy.bool
                    and mask.shape == (n,)
                ), (mask, str(mask.device()))
                assert isinstance(info, np.ndarray | jax.Array | list), info
                for mask_i, info_i in zip(mask, info):
                    if mask_i:
                        if key == "final_observation":
                            assert info_i in env.single_observation_space, (
                                info_i,
                                type(info_i),
                                env.single_observation_space,
                            )
                        else:
                            assert info_i is not None
                    else:
                        assert info_i is None
        assert "_final_observation" in infos
        assert "final_observation" in infos
        assert "_final_info" in infos
        assert "final_info" in infos


def _check_episode_tensor(
    v: torch.Tensor | jax.Array,
    device: torch.device,
    space: gymnasium.Space[Tensor] | None = None,
    nested: bool = False,
    dtype: torch.dtype | None = None,
):
    if isinstance(v, jax.Array):
        v = jax_to_torch_tensor(v)

    assert isinstance(v, Tensor) and v.device == device, v
    assert not v.is_nested
    if space:
        assert all(v_i in space for v_i in v), (len(v), v[0], space)
    if dtype:
        assert v.dtype == dtype


def check_episode(episode: Episode, env: gymnasium.Env[Tensor, Any], device: torch.device):
    assert episode["observations"] is episode.observations

    if isinstance(env, VectorEnv):
        observation_space = env.single_observation_space
    else:
        observation_space = env.observation_space
    _check_episode_tensor(episode.observations, device=device, space=observation_space)

    assert episode["actions"] is episode.actions
    if isinstance(env, VectorEnv):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
    _check_episode_tensor(episode.actions, device=device, space=action_space)

    assert episode["rewards"] is episode.rewards
    assert isinstance(episode.rewards, jax.Array | torch.Tensor)
    rewards = episode.rewards
    if isinstance(rewards, jax.Array):
        rewards = jax_to_torch_tensor(rewards)
    _check_episode_tensor(rewards, device=device, dtype=torch.float32)

    assert episode["terminated"] is episode.terminated
    assert isinstance(episode.terminated, bool | torch.Tensor | jax.Array), episode.terminated
    if isinstance(episode.terminated, torch.Tensor | jax.Array):
        assert not episode.terminated.shape
    if isinstance(episode.terminated, torch.Tensor):
        assert episode.terminated.device == device
        assert episode.terminated.dtype == torch.bool

    assert episode["truncated"] is episode.truncated
    assert isinstance(episode.truncated, bool | torch.Tensor | jax.Array), episode.truncated
    if isinstance(episode.truncated, torch.Tensor | jax.Array):
        assert not episode.truncated.shape
    if isinstance(episode.truncated, torch.Tensor):
        assert episode.truncated.device == device
        assert episode.truncated.dtype == torch.bool


def check_episode_batch(
    episode: EpisodeBatch, env: VectorEnv[Tensor, Any], batch_size: int, device: torch.device
):
    obs = episode.observations
    assert isinstance(obs, Tensor) and obs.device == device

    def _check_episode_batch_tensor(
        v: Tensor,
        single_space: TensorBox | TensorDiscrete | gymnasium.Space[Tensor] | None = None,
        dtype: torch.dtype | None = None,
    ):
        shape = get_shape_ish(v)
        assert shape[0] == batch_size
        if v.is_nested:
            assert shape[1] == "?"
        elif len(shape) > 1:
            assert isinstance(shape[1], int)
        if single_space:
            assert shape[2:] == single_space.shape
            if dtype is None:
                dtype = single_space.dtype
        assert v.device == device
        if dtype:
            assert v.dtype == dtype

    _check_episode_batch_tensor(obs, single_space=env.single_observation_space)
    _check_episode_batch_tensor(episode.actions, single_space=env.single_action_space)
    _check_episode_batch_tensor(episode.rewards, dtype=torch.float32)
    _check_episode_batch_tensor(episode.terminated, dtype=torch.bool)
    _check_episode_batch_tensor(episode.truncated, dtype=torch.bool)

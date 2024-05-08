from collections.abc import Mapping
from typing import Any

import gymnasium
import pytest
import torch
import torch.utils
import torch.utils.data
from torch import Tensor
from torch.utils.data import DataLoader

from project.datamodules.rl.envs import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.rl_datamodule import custom_collate_fn
from project.datamodules.rl.rl_dataset import RlDataset, VectorEnvRlDataset
from project.datamodules.rl.rl_types import (
    Env,
    Episode,
    EpisodeBatch,
    MappingMixin,
    VectorEnv,
    random_actor,
)
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete
from project.utils.types import NestedDict, NestedMapping
from project.utils.utils import get_shape_ish


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
def vector_env(env_id: str, seed: int, num_envs: int, device: torch.device):
    return make_torch_vectorenv(env_id=env_id, num_envs=num_envs, seed=seed, device=device)


pytest.register_assert_rewrite(__file__)


def _check_episode_tensor(
    v: Any,
    device: torch.device,
    space: gymnasium.Space[Tensor] | None = None,
    nested: bool = False,
    dtype: torch.dtype | None = None,
):
    assert isinstance(v, Tensor) and v.device == device
    assert not v.is_nested
    if space:
        assert all(v_i in space for v_i in v), (len(v), v[0], space)
    if dtype:
        assert v.dtype == dtype


def _check_episode(episode: Episode, env: Env[Tensor, Any], device: torch.device):
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
    _check_episode_tensor(episode.rewards, device=device, dtype=torch.float32)

    assert episode["terminated"] is episode.terminated
    _check_episode_tensor(episode.terminated, device=device, dtype=torch.bool)

    assert episode["truncated"] is episode.truncated
    _check_episode_tensor(episode.truncated, device=device, dtype=torch.bool)


def _check_episode_batch(
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


@pytest.mark.timeout(300)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "halfcheetah"], indirect=True)
def test_rl_dataset(env: Env[Tensor, Tensor], seed: int, device: torch.device):
    episodes_per_epoch = 2
    dataset = RlDataset(env, actor=random_actor, episodes_per_epoch=episodes_per_epoch, seed=seed)
    for episode_index, episode in enumerate(dataset):
        assert isinstance(episode, Episode)
        _check_episode(episode, env=env, device=device)

    assert episode_index == episodes_per_epoch - 1


@pytest.mark.timeout(300)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "halfcheetah"], indirect=True)
def test_vectorenv_rl_dataset(
    vector_env: VectorEnv[Tensor, Tensor], seed: int, device: torch.device
):
    episodes_per_epoch = 3
    dataset = VectorEnvRlDataset(
        vector_env, actor=random_actor, episodes_per_epoch=episodes_per_epoch, seed=seed
    )
    for episode_index, episode in enumerate(dataset):
        assert isinstance(episode, Episode)
        _check_episode(episode, env=vector_env, device=device)
        assert episode_index < episodes_per_epoch

    assert episode_index == episodes_per_epoch - 1


@pytest.mark.timeout(600)
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "halfcheetah"], indirect=True)
def test_vectorenv_rl_dataset_with_dataloader(
    vector_env: VectorEnv[Tensor, Tensor], seed: int, device: torch.device
):
    episodes_per_epoch = 4
    dataset = VectorEnvRlDataset(
        vector_env, actor=random_actor, episodes_per_epoch=episodes_per_epoch, seed=seed
    )
    batch_size = 2
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, collate_fn=custom_collate_fn
    )
    for batch_index, episode_batch in enumerate(dataloader):
        assert isinstance(episode_batch, EpisodeBatch)
        _check_episode_batch(episode_batch, vector_env, batch_size=batch_size, device=device)
        assert batch_index < episodes_per_epoch // batch_size
    assert batch_index == (episodes_per_epoch // batch_size) - 1


def devices_dict(d: NestedMapping[str, Tensor | Any]) -> NestedDict[str, torch.device | None]:
    result: NestedDict[str, torch.device | None] = {}
    for k, v in d.items():
        if isinstance(v, Mapping | dict | MappingMixin):
            result[k] = devices_dict(v)
        else:
            result[k] = getattr(v, "device", None)
    return result

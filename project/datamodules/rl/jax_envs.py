import dataclasses
from typing import Any

import brax
import brax.envs
import gymnax
import jax
import torch
from brax.generalized.base import State
from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper
from jax import dlpack as jax_dlpack
from torch import Tensor
from torch.utils import dlpack as torch_dlpack

from project.datamodules.rl.gym_utils import ToTensorsWrapper
from project.utils.device import default_device

from .gym_utils import to_tensor


@to_tensor.register(jax.Array)
def _jax_tensor_to_torch_tensor(
    value: jax.Array, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    dpack = jax_dlpack.to_dlpack(value)  # type: ignore
    tensor = torch_dlpack.from_dlpack(dpack)
    return torch.as_tensor(tensor, dtype=dtype, device=device)


@to_tensor.register(State)
def _brax_state_to_torch(
    value: State, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> dict[str, Tensor | Any]:
    # NOTE: jax_to_torch returns None when it doesn't support a data type?!
    # return jax_to_torch(value, device=device)
    return to_tensor(dataclasses.asdict(value), dtype=dtype, device=device)


def gymnax_env(env_id: str, device: torch.device = default_device(), seed: int = 123):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToGymWrapper(gymnax_env, params=env_params, seed=seed)
    env = ToTensorsWrapper(env, device=device)
    return env


def brax_env(env_id: str, device: torch.device = default_device(), seed: int = 123, **kwargs):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, **kwargs)
    from brax.envs.wrappers.torch import TorchWrapper  # noqa
    from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper  # noqa
    from brax.io.torch import jax_to_torch  # noqa

    env = GymWrapper(brax_env, seed=seed)
    # env = TorchWrapper(env, device=device)
    from gymnasium.wrappers.compatibility import EnvCompatibility

    env = EnvCompatibility(env)
    env = ToTensorsWrapper(env, device=device, from_jax=True)
    return env


def gymnax_vectorenv(
    env_id: str, num_envs: int = 4096, device: torch.device = default_device(), seed: int = 123
):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToVectorGymWrapper(gymnax_env, num_envs=num_envs, params=env_params, seed=seed)
    env = ToTensorsWrapper(env, device=device)
    return env

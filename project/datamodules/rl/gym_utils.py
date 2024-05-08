from __future__ import annotations

from logging import getLogger as get_logger

import brax.envs.wrappers.gym
import gym
import gym.spaces
import gymnasium
import gymnax
import gymnax.wrappers
import torch

from project.datamodules.rl.jax_envs import brax_env, brax_vectorenv, gymnax_env, gymnax_vectorenv
from project.datamodules.rl.rl_types import BoxSpace, _Env
from project.datamodules.rl.wrappers.normalize_actions import NormalizeBoxActionWrapper
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox
from project.datamodules.rl.wrappers.to_tensor import ToTorchWrapper

logger = get_logger(__name__)


def check_and_normalize_box_actions(env: _Env) -> _Env:
    """Wrap env to normalize actions if [low, high] != [-1, 1]."""
    if isinstance(env.action_space, BoxSpace | TensorBox):
        low, high = env.action_space.low, env.action_space.high
        if (low != -1).any() or (high != 1).any():
            logger.info("Normalizing environment actions.")
            return NormalizeBoxActionWrapper(env)
    # Environment does not need to be normalized.
    return env


# def tuple_to_torch(
#     value: dict[str, Any], dtype: torch.dtype | None = None, device: torch.device | None = None
# ) -> tuple[torch.Tensor | Any, ...]:
#     """Converts a dict of jax.Arrays into a dict of PyTorch tensors."""
#     return type(value)(to_tensor(v, dtype=dtype, device=device) for v in value)  # type: ignore


# def __repr__(self) -> str:
#     class_name = type(self).__name__
#     if self.start != 0:
#         return f"{class_name}({self.n}, start={self.start}, device={self.device})"
#     return f"{class_name}({self.n}, device={self.device})"


def make_torch_env(env_id: str, seed: int, device: torch.device, **kwargs):
    if env_id in gymnax.registered_envs:
        return gymnax_env(env_id=env_id, seed=seed, device=device, **kwargs)
    if env_id in brax.envs._envs:
        return brax_env(env_id, device=device, seed=seed, **kwargs)

    env = gym.make(env_id, **kwargs)
    return ToTorchWrapper(env, device=device)


def make_torch_vectorenv(env_id: str, num_envs: int, seed: int, device: torch.device, **kwargs):
    if env_id in gymnax.registered_envs:
        return gymnax_vectorenv(
            env_id=env_id, num_envs=num_envs, seed=seed, device=device, **kwargs
        )
    if env_id in brax.envs._envs:
        return brax_vectorenv(env_id, num_envs=num_envs, seed=seed, device=device, **kwargs)
    env = gymnasium.vector.make(env_id, num_envs=num_envs, **kwargs)
    return ToTorchWrapper(env, device=device)

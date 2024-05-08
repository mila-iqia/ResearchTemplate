import collections
import collections.abc
import functools
from typing import Any, SupportsFloat

import brax
import brax.envs
import brax.envs.wrappers.gym
import gym.spaces
import gymnasium
import gymnax
import gymnax.wrappers
import jax
import numpy as np
import torch
from brax.io.torch import torch_to_jax
from gymnasium import Wrapper
from gymnasium.wrappers.compatibility import EnvCompatibility
from torch import Tensor
from typing_extensions import Generic, TypeVar  # noqa

from project.datamodules.rl.rl_types import (
    BoxSpace,
    VectorEnv,
    VectorEnvWrapper,
    _Env,
)
from project.datamodules.rl.wrappers import jax_torch_interop
from project.utils.types import NestedDict


@functools.singledispatch
def to_torch(
    value: Any, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Any:
    raise NotImplementedError(f"No handler for values of type {type(value)}")


@to_torch.register(torch.Tensor | np.ndarray | int | float | bool)
def _to_tensor(
    value: Any, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


@to_torch.register(type(None))
def _no_op[T](
    value: T, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> T:
    return value


@to_torch.register(collections.abc.Mapping)
def dict_to_torch(
    value: dict[str, Any], *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> dict[str, torch.Tensor | Any]:
    """Converts a dict of jax.Arrays into a dict of PyTorch tensors."""
    return type(value)(**{k: to_torch(v, dtype=dtype, device=device) for k, v in value.items()})  # type: ignore


def _jax_to_torch_with_dtype_and_device(
    value: jax.Array, *, dtype: torch.dtype | None = None, device: torch.device | None = None
):
    torch_val = jax_torch_interop.jax_to_torch_tensor(value)
    # avoiding transferring between devices, should already be on the right device!
    assert device is None or (torch_val.device == device)
    return torch_val.to(dtype=dtype)


def wrapped_env_is_jax(env: _Env) -> bool:
    if isinstance(env, gym.Wrapper | gymnasium.Wrapper):
        env = env.unwrapped
    if isinstance(env, EnvCompatibility):
        # For some reason EnvCompatibility doesn't allow `unwrapped` to get to the base legacy env.
        env = env.env  # type: ignore
    if isinstance(env, gym.Wrapper | gymnasium.Wrapper):
        env = env.unwrapped
    return isinstance(
        env,
        gymnax.wrappers.GymnaxToGymWrapper
        | gymnax.wrappers.GymnaxToVectorGymWrapper
        | brax.envs.wrappers.gym.GymWrapper,
    )


TensorObsType = TypeVar(
    "TensorObsType", bound=Tensor | NestedDict[str, Tensor], default=torch.Tensor
)
TensorActType = TypeVar(
    "TensorActType", bound=Tensor | NestedDict[str, Tensor], default=torch.Tensor
)


class ToTorchWrapper(Wrapper[TensorObsType, TensorActType, Any, Any]):
    """Wrapper that moves numpy arrays to torch tensors and vice versa.

    Very bad, very sad. This is only useful because it moves stuff directly to GPU, but there's a
    high performance cost to doing this. Only use this if you were going to do this anyway because
    your env is returning numpy arrays or similar. Consider using an environment that natively runs
    on the GPU.
    """

    def __init__(
        self,
        env: gymnasium.Env[Any, Any],
        device: torch.device,
    ):
        super().__init__(env)
        self.device = device
        assert isinstance(env.observation_space, BoxSpace), (
            env.observation_space,
            type(env.observation_space),
        )
        self.observation_space = to_torch(env.observation_space, device=self.device)
        self.action_space = to_torch(env.action_space, device=self.device)

    def reset(
        self, seed: int | None = None, options: dict | None = None, **kwargs
    ) -> tuple[TensorObsType, dict]:
        """Resets the environment, returning a modified observation using
        :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self.observation(obs), self.info(info)

    def step(
        self, action: TensorActType
    ) -> tuple[TensorObsType, SupportsFloat, torch.BoolTensor, torch.BoolTensor, dict]:
        """Returns a modified observation using :meth:`self.observation` after calling
        :meth:`env.step`."""
        np_action = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(np_action)
        observation = self.observation(observation)
        reward = self.reward(reward)
        terminated = to_torch(terminated, dtype=torch.bool, device=self.device)
        truncated = to_torch(truncated, dtype=torch.bool, device=self.device)
        info = self.info(info)
        return observation, reward, terminated, truncated, info

    def observation(self, observation: np.ndarray) -> TensorObsType:
        return to_torch(observation, dtype=self.observation_space.dtype, device=self.device)  # type: ignore

    def info(self, info: dict[str, Any]) -> dict[str, Tensor | Any]:
        # if self.wrapped_env_is_jax:
        # return dict_to_torch(info, device=self.device)
        # By default we don't do anything with the info dict.
        return info

    def action(self, action: TensorActType) -> np.ndarray | jax.numpy.ndarray:
        if self.wrapped_env_is_jax:
            return torch_to_jax(action)
        return action.detach().cpu().numpy()

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return to_torch(reward, dtype=torch.float32, device=self.device)


class ToTorchVectorEnvWrapper(ToTorchWrapper, VectorEnvWrapper[Tensor, Tensor, Any, Any]):
    def __init__(self, env: VectorEnv[Any, Any], device: torch.device):
        super().__init__(env, device=device)
        self.single_observation_space = to_torch(env.single_observation_space, device=self.device)
        self.single_action_space = to_torch(env.single_action_space, device=self.device)

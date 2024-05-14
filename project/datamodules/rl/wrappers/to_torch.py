import collections
import collections.abc
import copy
import functools
from typing import Any, SupportsFloat

import gymnasium
import numpy as np
import torch
from gymnasium import Wrapper
from torch import Tensor

from project.datamodules.rl.rl_types import (
    BoxSpace,
    VectorEnv,
    VectorEnvWrapper,
)

from .tensor_spaces import TensorBox, TensorDiscrete, TensorSpace


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


class ToTorchWrapper(Wrapper[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]):
    """Wrapper that moves numpy arrays to torch tensors and vice versa.

    Prefer using a dedicated Jax to Torch wrapper like the ones for Brax and Gymnax instead of this
    one, since we want to avoid unintentionally moving stuff between devices.

    This is only useful because it moves stuff directly to GPU, but there's a high performance cost
    to doing this. Only use this if you were going to do this anyway because your env returns numpy
    arrays or similar. Consider using an environment that natively runs on the GPU.
    """

    def __init__(
        self,
        env: gymnasium.Env[np.ndarray, np.ndarray],
        device: torch.device,
    ):
        super().__init__(env)
        self.device = device
        assert isinstance(env.observation_space, BoxSpace), (
            env.observation_space,
            type(env.observation_space),
        )
        self.observation_space: TensorBox = to_torch(env.observation_space, device=self.device)
        self.action_space: TensorSpace = to_torch(env.action_space, device=self.device)

    def reset(
        self, seed: int | None = None, options: dict | None = None, **kwargs
    ) -> tuple[Tensor, dict]:
        """Resets the environment, returning a modified observation using
        :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self.observation(obs), self.info(info)

    def step(
        self, action: Tensor
    ) -> tuple[Tensor, Tensor, torch.BoolTensor, torch.BoolTensor, dict]:
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

    def observation(self, observation: np.ndarray) -> Tensor:
        return torch.as_tensor(observation, dtype=self.observation_space.dtype, device=self.device)

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        # By default we don't do anything with the info dict.
        return info

    def action(self, action: Tensor) -> np.ndarray:
        numpy_action = action.detach().cpu().numpy()
        assert numpy_action in self.env.action_space
        return numpy_action

    def reward(self, reward: SupportsFloat) -> Tensor:
        return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@to_torch.register(gymnasium.spaces.Box)
def gymnasium_box_to_tensor_space(
    space: gymnasium.spaces.Box,
    /,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TensorBox:
    return TensorBox(
        low=space.low,
        high=space.high,
        shape=space.shape,
        dtype=dtype,
        device=device or torch.device("cpu"),
    )


@to_torch.register(gymnasium.spaces.Discrete)
def gymnasium_discrete_to_tensor_space(
    space: gymnasium.spaces.Discrete,
    /,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TensorDiscrete:
    return TensorDiscrete(
        start=int(space.start),
        n=int(space.n),
        dtype=dtype or torch.int64,
        device=device or torch.device("cpu"),
    )


@to_torch.register(gymnasium.spaces.Tuple)
def _gymnasium_tuple_space_to_tensor(
    space: gymnasium.spaces.Tuple,
    /,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> gymnasium.spaces.Tuple:
    return type(space)(
        [to_torch(subspace, dtype=dtype, device=device) for subspace in space.spaces],
        seed=copy.deepcopy(space.np_random),
    )


@to_torch.register(gymnasium.spaces.Dict)
def _gymnasium_dict_space_to_tensor(
    space: gymnasium.spaces.Dict,
    /,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> gymnasium.spaces.Dict:
    return type(space)(
        {
            key: to_torch(subspace, dtype=dtype, device=device)
            for key, subspace in space.spaces.items()
        },
    )


class ToTorchVectorEnvWrapper(ToTorchWrapper, VectorEnvWrapper[Tensor, Tensor, Any, Any]):
    def __init__(self, env: VectorEnv[Any, Any], device: torch.device):
        super().__init__(env, device=device)
        self.single_observation_space = to_torch(env.single_observation_space, device=self.device)
        self.single_action_space = to_torch(env.single_action_space, device=self.device)

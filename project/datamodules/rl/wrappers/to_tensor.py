import collections
import collections.abc
import dataclasses
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
from brax.generalized.base import State
from brax.io.torch import torch_to_jax
from gymnasium.wrappers.compatibility import EnvCompatibility
from jax import dlpack as jax_dlpack
from torch import Tensor
from torch.utils import dlpack as torch_dlpack
from typing_extensions import Generic, TypeVar  # noqa

from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete
from project.utils.types import NestedDict

from ..rl_types import BoxSpace, DiscreteSpace, VectorEnv, VectorEnvWrapper, Wrapper, _Env


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


@to_torch.register(jax.Array)
def _jax_tensor_to_torch_tensor(
    value: jax.Array, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    dpack = jax_dlpack.to_dlpack(value)  # type: ignore
    tensor = torch_dlpack.from_dlpack(dpack)
    return torch.as_tensor(tensor, dtype=dtype, device=device)


@to_torch.register(State)
def _brax_state_to_torch(
    value: State, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> dict[str, Tensor | Any]:
    # NOTE: jax_to_torch returns None when it doesn't support a data type?!
    # return jax_to_torch(value, device=device)
    return to_torch(dataclasses.asdict(value), dtype=dtype, device=device)


@to_torch.register(BoxSpace)
def _box_space(
    value: BoxSpace, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> TensorBox:
    return TensorBox(  # type: ignore
        low=value.low,
        high=value.high,
        dtype=dtype if dtype is not None else value.dtype,
        device=device,
    )


@to_torch.register(DiscreteSpace)
def _discrete_space(
    value: DiscreteSpace, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> TensorDiscrete:
    return TensorDiscrete(  # type: ignore
        n=value.n,
        start=value.start,
        dtype=value.dtype,
        # seed=env.seed,
    )


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
    """TODO: Unclear if this wrapper should be from numpy to Torch only, or also include jax to torch.."""

    def __init__(
        self,
        env: gymnasium.Env[Any, Any],
        device: torch.device,
        from_jax: bool | None = None,
    ):
        super().__init__(env)
        if from_jax is None:
            from_jax = wrapped_env_is_jax(env)
        self.wrapped_env_is_jax = from_jax
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
        brax_env: brax.envs.wrappers.gym.GymWrapper | None = None
        if isinstance(self.env, EnvCompatibility) and isinstance(
            self.env.env, brax.envs.wrappers.gym.GymWrapper
        ):
            brax_env = self.env.env
        elif isinstance(self.env, brax.envs.wrappers.gym.GymWrapper):
            brax_env = self.env

        if brax_env is not None and brax_env._state is not None:
            # Need to slightly adjust the reset of the wrapped brax env to take in a seed.
            # Here is the code of the `GymWrapper.reset` at the time of writing:
            # Can actually get the reset info from the env:
            info = {**brax_env._state.metrics, **brax_env._state.info}

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

    # if typing.TYPE_CHECKING:
    #     observation_space: TensorDiscrete | TensorBox

    def observation(self, observation: np.ndarray) -> TensorObsType:
        return to_torch(observation, dtype=self.observation_space.torch_dtype, device=self.device)  # type: ignore

    def info(self, info: dict[str, Any]) -> dict[str, Tensor | Any]:
        if self.wrapped_env_is_jax:
            return dict_to_torch(info, device=self.device)
        return info

    def action(self, action: TensorActType) -> np.ndarray | jax.numpy.ndarray:
        if self.wrapped_env_is_jax:
            return torch_to_jax(action)
        return action.detach().cpu().numpy()

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return to_torch(reward, dtype=torch.float32, device=self.device)


class ToTorchVectorEnvWrapper(ToTorchWrapper, VectorEnvWrapper[Tensor, Tensor, Any, Any]):
    def __init__(
        self, env: VectorEnv[Any, Any], device: torch.device, from_jax: bool | None = None
    ):
        super().__init__(env, device=device, from_jax=from_jax)
        self.single_observation_space = to_torch(env.single_observation_space, device=self.device)
        self.single_action_space = to_torch(env.single_action_space, device=self.device)

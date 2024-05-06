import functools
from collections.abc import Callable
from typing import Any, ClassVar, Concatenate

import brax.envs.wrappers.gym
import brax.generalized.base
import brax.io.image
import gym.spaces
import gymnasium
import gymnasium.core
import gymnasium.spaces.utils
import gymnax
import gymnax.environments.spaces
import jax
import numpy as np
import torch
from brax.envs.base import State
from brax.envs.wrappers.gym import GymWrapper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.jax_torch_interop import (
    jax_array_to_torch_tensor,
    jax_to_torch,
    torch_tensor_to_jax_array,
)
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    get_torch_dtype,
    get_torch_dtype_from_jax_dtype,
)
from project.datamodules.rl.wrappers.to_tensor import ToTorchVectorEnvWrapper
from project.utils.device import default_device


def gymnax_env(env_id: str, device: torch.device = default_device(), seed: int = 123):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToGymWrapper(gymnax_env, params=env_params, seed=seed)
    env = GymnaxToTorchWrapper(env, device=device)
    # NOTE: The spaces are also seeded here.
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    # env = ToTorchWrapper(env, device=device)
    # env.observation_space.seed(seed)
    # env.action_space.seed(seed)
    return env


def brax_env(env_id: str, device: torch.device = default_device(), seed: int = 123, **kwargs):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, **kwargs)
    env = GymWrapper(brax_env, seed=seed)
    env = BraxToTorchWrapper(env)
    # env = ToTorchWrapper(env, device=device, from_jax=True)
    return env


def gymnax_vectorenv(
    env_id: str, num_envs: int = 4096, device: torch.device = default_device(), seed: int = 123
):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToVectorGymWrapper(gymnax_env, num_envs=num_envs, params=env_params, seed=seed)
    env = ToTorchVectorEnvWrapper(env, device=device)
    raise NotImplementedError("todo")
    return env


def brax_vectorenv(
    env_id: str,
    num_envs: int = 4096,
    device: torch.device = default_device(),
    seed: int = 123,
    **kwargs,
):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, batch_size=num_envs, **kwargs)
    env = VectorGymnasiumWrapper(brax_env, seed=seed)
    # todo: Not 100% sure that this EnvCompatibility wrapper can be used on a VectorEnv.
    # env = VectorEnvCompatibility(env)  # make the env compatible with the gymnasium api
    env = ToTorchVectorEnvWrapper(env, device=device, from_jax=True)
    return env


class JaxToTorchWrapper(gymnasium.Wrapper[torch.Tensor, torch.Tensor, jax.Array, jax.Array]):
    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, dict[Any, Any]
    ]:
        jax_action = torch_tensor_to_jax_array(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)
        torch_obs = jax_array_to_torch_tensor(obs)
        assert isinstance(reward, jax.Array)
        torch_reward = jax_array_to_torch_tensor(reward)
        assert isinstance(terminated, jax.Array)
        torch_terminated = jax_array_to_torch_tensor(terminated)
        assert isinstance(truncated, jax.Array)
        torch_truncated = jax_array_to_torch_tensor(truncated)
        torch_info = jax_to_torch(info)

        # debug: checking that the devices are the same for everything, so that we don't have to
        # move stuff.
        jax_devices = jax_action.devices()
        assert reward.devices() == jax_devices
        assert terminated.devices() == jax_devices
        assert truncated.devices() == jax_devices

        return torch_obs, torch_reward, torch_terminated, torch_truncated, torch_info  # type: ignore

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        obs, info = self.env.reset(seed=seed, options=options)
        torch_obs = jax_array_to_torch_tensor(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info


class GymnaxToTorchWrapper(JaxToTorchWrapper):
    def __init__(
        self,
        env: GymnaxToGymWrapper[jax.Array, jax.Array],
        device: torch.device,
    ):
        super().__init__(env=env)
        self.device = device
        self.env: GymnaxToGymWrapper[jax.Array, jax.Array]

        observation_space = self.env._env.observation_space(self.env_params)
        action_space = self.env._env.action_space(self.env_params)

        self.observation_space = gymnax_space_to_torch_space(observation_space)
        self.action_space = gymnax_space_to_torch_space(action_space)

    def render(self) -> gymnasium.core.RenderFrame | list[gymnasium.core.RenderFrame] | None:
        """use underlying environment rendering if it exists, otherwise return None."""
        return self.env.render()


class BraxToTorchWrapper(JaxToTorchWrapper):
    def __init__(
        self,
        env: brax.envs.wrappers.gym.GymWrapper,
    ):
        brax_wrapper = env
        new_style_env = EnvCompatibility(env)  # make the env compatible with the newer Gym api
        super().__init__(env=new_style_env)  # type: ignore
        assert isinstance(brax_wrapper.observation_space, gym.spaces.Box)
        assert isinstance(brax_wrapper.action_space, gym.spaces.Box)
        _state = brax_wrapper._env.reset(jax.random.key(0))

        device = get_torch_device_from_jax_array(_state.obs)

        # BUG: The observation space uses np.nan, which is bad!
        # Seems like we could use the DoF to set the limits here:
        # https://github.com/google/brax/issues/360#issuecomment-1585227475
        _min, _max = brax_wrapper._env.sys.dof.limit
        # TODO: Figure out how to use the sys info to get the observation space upper/lower limits.
        assert _min.shape == _max.shape == _state.obs.shape, (
            _min.shape,
            _max.shape,
            _state.obs.shape,
        )
        assert (_min <= _state.obs).all()
        assert (_state.obs <= _max).all()

        self.observation_space = TensorBox(
            low=(low := jax_array_to_torch_tensor(_min)),
            high=jax_array_to_torch_tensor(_max),
            shape=_state.obs.shape,
            dtype=low.dtype,
            device=device,
        )
        assert not np.isnan(brax_wrapper.action_space.low).any()
        assert not np.isnan(brax_wrapper.action_space.high).any()
        action_dtype = get_torch_dtype(brax_wrapper.action_space.dtype)
        self.action_space = TensorBox(
            low=torch.as_tensor(brax_wrapper.action_space.low, dtype=action_dtype, device=device),
            high=torch.as_tensor(
                brax_wrapper.action_space.high, dtype=action_dtype, device=device
            ),
            shape=brax_wrapper.action_space.shape,
            dtype=action_dtype,
            device=device,
        )

    def render(self):
        assert self.env.render_mode == "rgb_array"
        # TODO: check what this actually returns.
        return self.env.render()


@functools.singledispatch
def to_torch_space(
    space: Any, /, *, device: torch.device | None = None
) -> gymnasium.Space[torch.Tensor]:
    raise NotImplementedError(f"No handler for values of type {type(space)}")


@to_torch_space.register(gymnasium.spaces.Box)
@to_torch_space.register(gym.spaces.Box)
def _box_to_torch_box(
    space: gymnasium.spaces.Box | gym.spaces.Box, /, *, device: torch.device | None = None
) -> TensorBox:
    return TensorBox(
        low=space.low,
        high=space.high,
        shape=space.shape,
        dtype=get_torch_dtype(space.dtype),
        device=device or default_device(),
    )


@to_torch_space.register(gymnasium.spaces.Discrete)
@to_torch_space.register(gym.spaces.Discrete)
def _discrete_to_torch_discrete(
    space: gymnasium.spaces.Discrete | gym.spaces.Discrete,
    /,
    *,
    device: torch.device | None = None,
) -> TensorDiscrete:
    return TensorDiscrete(
        n=int(space.n),
        start=int(space.start),
        dtype=get_torch_dtype(space.dtype),  # type: ignore
        device=device or default_device(),
    )


@to_torch_space.register(gymnax.environments.spaces.Space)
@functools.singledispatch
def gymnax_space_to_torch_space(
    gymnax_space: gymnax.environments.spaces.Space, /, *, device: torch.device | None = None
) -> gymnasium.Space:
    raise NotImplementedError(f"No handler for gymnax spaces of type {type(gymnax_space)}")


def get_torch_device_from_jax_array(array: jax.Array) -> torch.device:
    jax_device = array.devices()
    assert len(jax_device) == 1
    jax_device_str = str(jax_device.pop())
    assert isinstance(jax_device_str, str), (jax_device_str, type(jax_device_str))
    if jax_device_str.startswith("cuda"):
        device_type, _, index = jax_device_str.partition(":")
        assert index.isdigit()
        return torch.device(device_type, int(index))
    return torch.device("cpu")


@gymnax_space_to_torch_space.register
def _gymnax_box_to_torch_box(
    gymnax_space: gymnax.environments.spaces.Box, /, *, device: torch.device | None = None
) -> TensorBox:
    if not device:
        jax_array = gymnax_space.sample(jax.random.key(0))
        assert isinstance(jax_array, jax.Array)
        device = get_torch_device_from_jax_array(jax_array)
    return TensorBox(
        low=jax_array_to_torch_tensor(gymnax_space.low)
        if isinstance(gymnax_space.low, jax.Array)
        else gymnax_space.low,
        high=jax_array_to_torch_tensor(gymnax_space.high)
        if isinstance(gymnax_space.high, jax.Array)
        else gymnax_space.high,
        shape=gymnax_space.shape,
        dtype=get_torch_dtype_from_jax_dtype(gymnax_space.dtype),
        device=device,
    )


@gymnax_space_to_torch_space.register
def _gymnax_discrete_to_torch_discrete(
    gymnax_space: gymnax.environments.spaces.Discrete, /, *, device: torch.device | None = None
) -> TensorDiscrete:
    if not device:
        jax_array = gymnax_space.sample(jax.random.key(0))
        assert isinstance(jax_array, jax.Array)
        device = get_torch_device_from_jax_array(jax_array)
    return TensorDiscrete(
        n=gymnax_space.n,
        start=0,
        dtype=get_torch_dtype_from_jax_dtype(gymnax_space.dtype),
        device=device,
    )


def jit[C: Callable, **P](
    c: C, _fn: Callable[Concatenate[C, P], Any] = jax.jit, *args: P.args, **kwargs: P.kwargs
) -> C:
    # Fix `jax.jit` so it preserves the jit-ed function's signature and docstring.
    return _fn(c, *args, **kwargs)


class VectorGymnasiumWrapper(VectorEnv[jax.Array, jax.Array]):
    """Adapt brax.envs.wrappers.gym.VectorGymWrapper for gymnasium compatibility.

    A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API.
    """

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: brax.envs.base.PipelineEnv, seed: int = 0, backend: str | None = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        self.render_mode = "rgb_array"
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size  # type: ignore
        self.seed(seed)
        self.backend = backend
        self._state: brax.envs.base.State | None = None

        obs = np.inf * np.ones(self._env.observation_size, dtype=np.float32)
        obs_space = gymnasium.spaces.Box(-obs, obs, dtype=np.float32)
        self.single_observation_space = obs_space
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        action_space = gymnasium.spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)
        self.single_action_space = action_space
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        def reset(key: jax.Array) -> tuple[State, jax.Array, jax.Array]:
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jit(reset, backend=self.backend)

        def step(
            state: State, action: jax.Array
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, dict[str, jax.Array | None]]:
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            # note: doesn't seem to be anything like 'truncated' in the brax state afaict.
            return state, state.obs, state.reward, state.done, info

        self._step = jit(step, backend=self.backend)

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict | None = None
    ) -> tuple[jax.Array, dict]:
        if isinstance(seed, list):
            key = jax.random.key(jax.numpy.asarray(seed))
        elif seed is not None:
            key = jax.random.key(seed)
        else:
            key = self._key
        self._state, obs, self._key = self._reset(key)
        info = {**self._state.metrics, **self._state.info}
        return obs, info

    def step(self, action: jax.Array):
        assert self._state is not None  # should have been reset first.
        self._state, obs, reward, done, info = self._step(self._state, action)

        return convert_to_terminated_truncated_step_api(
            (obs, reward, done, info),  # type: ignore
            is_vector_env=True,
        )

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self):
        assert self.render_mode == "rgb_array"
        sys, state = self._env.sys, self._state
        if state is None:
            raise RuntimeError("must call reset or step before rendering")
        # TODO: take a look at the `self._env.render()`
        # self._env.render(trajectory=[state.pipeline_state)
        # return brax.io.image.render_array(sys, state.pipeline_state.take(0), 256, 256)
        assert state.pipeline_state is not None
        return brax.io.image.render_array(sys, state.pipeline_state.take(0), 256, 256)

from collections.abc import Callable
from typing import Any, ClassVar, Concatenate

import brax
import brax.envs
import brax.envs.base
import brax.envs.wrappers.gym
import brax.generalized.base
import brax.io.image
import gymnasium
import gymnasium.spaces.utils
import gymnax
import jax
import numpy as np
import torch
from brax.envs.base import State
from brax.envs.wrappers.gym import GymWrapper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.to_tensor import ToTorchVectorEnvWrapper, ToTorchWrapper
from project.utils.device import default_device


def gymnax_env(env_id: str, device: torch.device = default_device(), seed: int = 123):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToGymWrapper(gymnax_env, params=env_params, seed=seed)
    env = ToTorchWrapper(env, device=device)
    return env


def brax_env(env_id: str, device: torch.device = default_device(), seed: int = 123, **kwargs):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, **kwargs)
    env = GymWrapper(brax_env, seed=seed)
    env = EnvCompatibility(env)  # make the env compatible with the newer Gym api
    env = ToTorchWrapper(env, device=device, from_jax=True)
    return env


def gymnax_vectorenv(
    env_id: str, num_envs: int = 4096, device: torch.device = default_device(), seed: int = 123
):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToVectorGymWrapper(gymnax_env, num_envs=num_envs, params=env_params, seed=seed)
    env = ToTorchVectorEnvWrapper(env, device=device)
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
        if self.render_mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("must call reset or step before rendering")
            assert state.pipeline_state is not None
            return brax.io.image.render_array(sys, state.pipeline_state.take(0), 256, 256)
        else:
            return super().render()  # just raise an exception

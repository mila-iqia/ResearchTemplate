import dataclasses
from logging import getLogger as get_logger
from typing import Any, ClassVar

import brax.envs
import brax.envs.wrappers.gym
import brax.envs.wrappers.training
import brax.training
import gym.spaces
import gymnasium
import gymnasium.envs.registration
import jax
import numpy as np
import torch
from brax.envs.base import State
from brax.envs.wrappers.gym import GymWrapper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.wrappers.compatibility import EnvCompatibility

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.jax_torch_interop import (
    JaxToTorchMixin,
    get_backend_from_torch_device,
    get_torch_device_from_jax_array,
    jax_to_torch,
    jax_to_torch_tensor,
    jit,
    torch_to_jax_tensor,
)
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, get_torch_dtype
from project.utils.device import default_device
from project.utils.types import NestedDict

logger = get_logger(__name__)


def brax_env(
    env_id: str,
    device: torch.device = default_device(),
    seed: int = 123,
    max_episode_steps: int = 1_000,
    **kwargs,
):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, episode_length=max_episode_steps, **kwargs)
    env = GymWrapper(
        brax_env,  # type: ignore (bad type hint in the brax wrapper constructor)
        seed=seed,
        backend=get_backend_from_torch_device(device),
    )
    env.spec.max_episode_steps = max_episode_steps
    env = BraxToTorchWrapper(env)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


class BraxToTorchWrapper(JaxToTorchMixin, gymnasium.Env[torch.Tensor, torch.Tensor]):
    """Compatibility fixes for the the GymWrapper of brax.

    1. It subclasses gym.Env, we'd like to be a subclass of gymnasium.Env
    2. It uses gym.spaces.Box for its obs / act spaces. We use gymnasium.spaces.Box.
    3. It follows the good old step API. We want to try to adopt the new gymnasium API.
    """

    def __init__(
        self,
        env: brax.envs.wrappers.gym.GymWrapper,
    ):
        super().__init__()
        self.brax_env = env
        # BUG: fix this, causes issues at the moment.
        max_episode_steps = env.spec.max_episode_steps
        self.spec = gymnasium.envs.registration.EnvSpec(
            **dataclasses.asdict(
                dataclasses.replace(
                    env.spec,
                    max_episode_steps=max_episode_steps,
                )
            )
        )
        # make the env compatible with the newer Gym api
        self.env = EnvCompatibility(env)  # type: ignore (expects a gymnasium.Env, gets gym.Env).
        assert isinstance(self.brax_env.observation_space, gym.spaces.Box)
        assert isinstance(self.brax_env.action_space, gym.spaces.Box)
        _state = self.brax_env._env.reset(jax.random.key(0))

        device = get_torch_device_from_jax_array(_state.obs)

        # BUG: The observation space uses np.nan, which is bad!
        # Seems like we could use the DoF to set the limits here:
        # https://github.com/google/brax/issues/360#issuecomment-1585227475
        # TODO: Figure out how to use the sys info to get the observation space upper/lower limits.
        _min, _max = self.brax_env._env.sys.dof.limit

        def _seed_or_none(space: gym.spaces.Space) -> int | None:
            if space._np_random is not None:
                return space._np_random.integers(0, 2**32).item()
            return None

        self.observation_space: TensorBox = TensorBox(
            low=(
                _low := torch.as_tensor(
                    self.brax_env.observation_space.low, dtype=torch.float32, device=device
                )
            ),
            high=torch.as_tensor(
                self.brax_env.observation_space.high, dtype=_low.dtype, device=device
            ),
            shape=_state.obs.shape,
            dtype=_low.dtype,
            device=device,
            seed=_seed_or_none(self.brax_env.observation_space),
        )
        action_dtype = get_torch_dtype(self.brax_env.action_space.dtype)
        self.action_space: TensorBox = TensorBox(
            low=torch.as_tensor(self.brax_env.action_space.low, dtype=action_dtype, device=device),
            high=torch.as_tensor(
                self.brax_env.action_space.high, dtype=action_dtype, device=device
            ),
            shape=self.brax_env.action_space.shape,
            dtype=action_dtype,
            device=device,
            seed=_seed_or_none(self.brax_env.action_space),
        )

    def reset(
        self, *, seed: int | None = None, options: NestedDict[str, Any] | None = None
    ) -> tuple[torch.Tensor, Any]:
        obs, info = self.env.reset(seed=seed, options=options)
        # the env doesn't return anything in info by default, but here I think it might be
        # useful to include the same information that is normally present in `step`:
        assert not info
        assert self.brax_env._state is not None
        info = {**self.brax_env._state.metrics, **self.brax_env._state.info}
        torch_obs = jax_to_torch_tensor(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info

    def render(self):
        assert self.env.render_mode == "rgb_array"
        # TODO: check what this actually returns.
        return self.env.render()


class BraxVectorEnvToTorchWrapper(JaxToTorchMixin, VectorEnv[torch.Tensor, torch.Tensor]):
    """Compatibility fixes for the the VectorGymWrapper of brax.

    1. It subclasses gym.vector.VectorEnv, we'd like to be a subclass of gymnasium.vector.VectorEnv
    2. It doesn't call super().__init__() with the num_envs and the single obs / act spaces. This
       makes it so the `single_observation_space` and `single_action_space` properties are not set.
    3. It uses gym.spaces.Box for its obs / act spaces. We use gymnasium.spaces.Box.
    """

    env: brax.envs.wrappers.gym.VectorGymWrapper

    def __init__(
        self, env: brax.envs.wrappers.gym.VectorGymWrapper, render_mode: str | None = None
    ):
        self.env = env
        self.metadata = env.metadata
        self.render_mode = render_mode
        self.reward_range = env.reward_range
        self.spec = env.spec  # type: ignore (todo: would have to convert the EnvSpec from one to the other...)
        # todo: would be better if we had a nicer way to know which device the env is running on.
        env.seed(123)
        _obs = env.reset()[0]
        device = get_torch_device_from_jax_array(_obs)
        obs_dtype = get_torch_dtype(_obs.dtype)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        # note: env.observation_space is the batched space in brax.
        assert env.observation_space.shape is not None
        single_observation_space = TensorBox(
            low=torch.as_tensor(env.observation_space.low[0], dtype=obs_dtype, device=device),
            high=torch.as_tensor(env.observation_space.high[0], dtype=obs_dtype, device=device),
            shape=env.observation_space.shape[1:],
            dtype=obs_dtype,
            device=device,
            seed=env.observation_space.np_random.integers(0, 2**32).item(),
        )
        assert isinstance(env.action_space.dtype, np.dtype)
        action_dtype = get_torch_dtype(env.action_space.dtype)
        assert env.action_space.shape is not None
        single_action_space = TensorBox(
            low=torch.as_tensor(env.action_space.low[0], dtype=action_dtype, device=device),
            high=torch.as_tensor(env.action_space.high[0], dtype=action_dtype, device=device),
            shape=env.action_space.shape[1:],
            dtype=action_dtype,
            device=device,
            seed=env.action_space.np_random.integers(0, 2**32).item(),
        )
        # NOTE: This is done by `VectorEnv`.
        # self.observation_space = gymnasium.vector.utils.batch_space(
        #     self.single_observation_space, env.num_envs
        # )
        # self.action_space = gymnasium.vector.utils.batch_space(
        #     self.single_action_space, env.num_envs
        # )

        super().__init__(
            num_envs=env.num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, dict[Any, Any]
    ]:
        jax_action = torch_to_jax_tensor(action)
        obs, reward, _dones, infos = self.env.step(jax_action)

        # Extracted from  gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api
        if isinstance(infos, list):
            _truncated = jax.numpy.array(
                [info.pop("TimeLimit.truncated", False) for info in infos]
            )
            terminated = (jax.numpy.logical_and(_dones, np.logical_not(_truncated)),)
            truncated = (jax.numpy.logical_and(_dones, _truncated),)
        elif isinstance(infos, dict):
            _num_envs = len(_dones)
            _truncated = infos.pop("TimeLimit.truncated", jax.numpy.zeros(_num_envs, dtype=bool))
            terminated = jax.numpy.logical_and(_dones, jax.numpy.logical_not(_truncated))
            truncated = jax.numpy.logical_and(_dones, _truncated)

        torch_obs = jax_to_torch_tensor(obs)
        assert isinstance(reward, jax.Array)
        torch_reward = jax_to_torch_tensor(reward)
        assert isinstance(terminated, jax.Array), (terminated, type(terminated))
        torch_terminated = jax_to_torch_tensor(terminated)
        assert isinstance(truncated, jax.Array)
        torch_truncated = jax_to_torch_tensor(truncated)
        # Brax has terminated and truncated as 0. and 1., here we convert them to bools instead.
        if torch_terminated.dtype != torch.bool:
            torch_terminated = torch_terminated.bool()
        if torch_truncated.dtype != torch.bool:
            torch_truncated = torch_truncated.bool()

        torch_info = jax_to_torch(infos)

        # debug: checking that the devices are the same for everything, so that we don't have to
        # move stuff.
        jax_devices = jax_action.devices()
        assert reward.devices() == jax_devices
        assert terminated.devices() == jax_devices
        assert truncated.devices() == jax_devices

        return torch_obs, torch_reward, torch_terminated, torch_truncated, torch_info  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: NestedDict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.env.seed(seed)
        assert not options  # don't know what to do with these..
        obs = self.env.reset()
        info = {}
        torch_obs = jax_to_torch_tensor(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        render_mode = self.render_mode or "rgb_array"
        return self.env.render(mode=render_mode)


def brax_vectorenv(
    env_id: str,
    num_envs: int = 4096,
    device: torch.device = default_device(),
    seed: int = 123,
    **kwargs,
):
    # Instantiate the environment & its settings.
    brax_env = brax.envs.create(env_id, batch_size=num_envs, **kwargs)
    brax_env = brax.envs.wrappers.gym.VectorGymWrapper(brax_env, seed=seed)
    # todo: Not 100% sure that this EnvCompatibility wrapper can be used on a VectorEnv.
    # env = VectorEnvCompatibility(env)  # make the env compatible with the gymnasium api
    env = BraxVectorEnvToTorchWrapper(brax_env)
    return env


class VectorGymnasiumWrapper(VectorEnv[jax.Array, jax.Array]):
    """Adapt brax.envs.wrappers.gym.VectorGymWrapper for gymnasium compatibility.

    A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API.

    TODO: Make a PR to add this to the Brax repo.
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

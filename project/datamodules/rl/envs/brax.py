import dataclasses
import functools
from logging import getLogger as get_logger
from typing import Any, ClassVar

import brax.envs
import brax.envs.wrappers.gym
import brax.envs.wrappers.training
import brax.io.image
import brax.training
import chex
import gym.spaces
import gymnasium
import gymnasium.envs.registration
import jax
import jax.numpy as jnp
import torch
from brax.envs.wrappers.gym import GymWrapper
from gymnasium.wrappers.compatibility import EnvCompatibility
from torch_jax_interop import (
    jax_to_torch,
    torch_to_jax,
)

from project.datamodules.rl.types import VectorEnv
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorSpace, get_torch_dtype
from project.utils.device import default_device
from project.utils.types import NestedDict
from project.utils.types.protocols import Dataclass

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
        backend=torch_to_jax(device).platform,
    )
    # Patch for the GymWrapper of brax
    env = BraxToTorchWrapper(env)
    # env.spec.max_episode_steps = max_episode_steps
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
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
    # In this case here, we need to completely revamp the way the `info` dict is created in `step`,
    # so we can't just patch `VectorGymWrapper`.

    # brax_env = brax.envs.wrappers.gym.VectorGymWrapper(brax_env, seed=seed)
    # todo: Not 100% sure that this EnvCompatibility wrapper can be used on a VectorEnv.
    # env = VectorEnvCompatibility(env)  # make the env compatible with the gymnasium api
    env = BraxToTorchVectorEnv(brax_env)
    return env



class _DataclassMeta(type):
    def __subclasscheck__(self, subclass: type) -> bool:
        return dataclasses.is_dataclass(subclass) and not dataclasses.is_dataclass(type(subclass))

    def __instancecheck__(self, instance: Any) -> bool:
        return dataclasses.is_dataclass(instance) and dataclasses.is_dataclass(type(instance))


# NOTE: Not using a `runtime_checkable` version of the `Dataclass` protocol here, because it
# doesn't work correctly in the case of `isinstance(SomeDataclassType, Dataclass)`, which returns
# `True` when it should be `False` (since it's a dataclass type, not a dataclass instance), and the
# runtime_checkable decorator doesn't check the type of the attribute (ClassVar vs instance
# attribute).
class _DataclassInstance(metaclass=_DataclassMeta): ...


@jax_to_torch.register(_DataclassInstance)
def jax_dataclass_to_torch_dataclass(value: Dataclass) -> NestedDict[str, torch.Tensor]:
    return jax_to_torch(dataclasses.asdict(value))


@torch_to_jax.register(_DataclassInstance)
def torch_dataclass_to_jax_dataclass(value: Dataclass) -> NestedDict[str, jax.Array]:
    return torch_to_jax(dataclasses.asdict(value))


class JaxToTorchMixin:
    """A mixin that just implements the step and reset that convert jax arrays into torch tensors.

    TODO: Eventually make this support dict / tuples observations and actions:
    - use the generic `jax_to_torch` function.
    - mark this class generic w.r.t. the type of observations and actions
    """

    env: gymnasium.Env[jax.Array, jax.Array] | VectorEnv[jax.Array, jax.Array]
    observation_space: TensorSpace
    action_space: TensorSpace

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        jax.Array,
        bool | jax.Array | torch.Tensor,
        bool | jax.Array | torch.Tensor,
        dict[Any, Any],
    ]:
        jax_action = torch_to_jax(
            action.contiguous() if not action.is_contiguous() else action
        )
        obs, reward, terminated, truncated, info = self.env.step(jax_action)
        torch_obs = jax_to_torch(obs)

        # IDEA: Keep the rewards as jax arrays, since most envs / wrappers of Gymnasium assume a jax array?
        assert isinstance(reward, jax.Array)
        assert isinstance(terminated, jax.Array | bool), terminated
        assert isinstance(truncated, jax.Array | bool), truncated
        # torch_reward = jax_to_torch_tensor(reward)
        # device = self.observation_space.device
        # if isinstance(terminated, bool):
        #     torch_terminated = torch.tensor(terminated, dtype=torch.bool, device=device)
        # else:
        #     assert isinstance(terminated, jax.Array)
        #     torch_terminated = jax_to_torch_tensor(terminated)

        # if isinstance(truncated, bool):
        #     torch_truncated = torch.tensor(truncated, dtype=torch.bool, device=device)
        # else:
        #     assert isinstance(truncated, jax.Array)
        #     torch_truncated = jax_to_torch_tensor(truncated)

        # Brax has terminated and truncated as 0. and 1., here we convert them to bools instead.
        if isinstance(terminated, jax.Array) and terminated.dtype != jnp.bool:
            terminated = terminated.astype(jnp.bool)

        if isinstance(truncated, jax.Array) and truncated.dtype != jnp.bool:
            truncated = truncated.astype(jnp.bool)

        torch_info = jax_to_torch(info)
        return torch_obs, reward, terminated, truncated, torch_info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        obs, info = self.env.reset(seed=seed, options=options)
        torch_obs = jax_to_torch(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info


class BraxToTorchWrapper(JaxToTorchMixin, gymnasium.Env[torch.Tensor, torch.Tensor]):
    """Compatibility fixes for the the GymWrapper of brax.

    1. It subclasses gym.Env, we'd like to be a subclass of gymnasium.Env
    2. It uses gym.spaces.Box for its obs / act spaces. We use TensorBox.
    3. It follows the good old step API. We want to try to adopt the new gymnasium API.
    """

    def __init__(
        self,
        env: brax.envs.wrappers.gym.GymWrapper,
        seed: int | None = None,
    ):
        super().__init__()
        self.brax_env = env
        # make the env compatible with the newer Gym api
        self.env = EnvCompatibility(env)  # type: ignore (expects a gymnasium.Env, gets gym.Env).
        assert isinstance(self.brax_env.observation_space, gym.spaces.Box)
        assert isinstance(self.brax_env.action_space, gym.spaces.Box)
        _state = self.brax_env._env.reset(jax.random.key(0))

        device = jax_to_torch(_state.obs).device

        # BUG: The observation space uses np.nan, which is bad!
        # Seems like we could use the DoF to set the limits here:
        # https://github.com/google/brax/issues/360#issuecomment-1585227475
        # TODO: Figure out how to use the sys info to get the observation space upper/lower limits.
        _min, _max = self.brax_env._env.sys.dof.limit

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
            seed=seed,
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
            seed=seed,
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
        torch_obs = jax_to_torch(obs)
        torch_info = jax_to_torch(info)
        return torch_obs, torch_info

    def render(self):
        assert self.env.render_mode == "rgb_array"
        # TODO: check what this actually returns.
        return self.env.render()


class BraxToTorchVectorEnv(VectorEnv[torch.Tensor, torch.Tensor]):
    """Compatibility fixes for the the VectorGymWrapper of brax.

    1. It subclasses gym.vector.VectorEnv, we'd like to be a subclass of gymnasium.vector.VectorEnv
    2. It doesn't call super().__init__() with the num_envs and the single obs / act spaces. This
       makes it so the `single_observation_space` and `single_action_space` properties are not set.
    3. It uses gym.spaces.Box for its obs / act spaces. We use gymnasium.spaces.Box.
    """

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(
        self,
        env: brax.envs.wrappers.training.AutoResetWrapper,
        seed: int = 0,
        render_mode: str = "rgb_array",
    ):
        self._env = env
        self.render_mode = render_mode
        self.is_vector_env = True
        self.closed = False
        self.viewer = None

        # Making things very strict for now, just so we fully understand everything that's
        # happening in each wrapper.
        assert isinstance(self._env, brax.envs.wrappers.training.AutoResetWrapper)
        assert isinstance(self._env.env, brax.envs.wrappers.training.VmapWrapper)
        assert isinstance(self._env.env.env, brax.envs.wrappers.training.EpisodeWrapper)
        assert isinstance(self._env.env.env.env, brax.envs.PipelineEnv)
        assert isinstance(self._env.unwrapped, brax.envs.PipelineEnv)
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.unwrapped.dt,
        }
        assert self._env.env.batch_size is not None
        self.num_envs = self._env.env.batch_size

        self._rng_key = jax.random.key(seed)
        self._state = None

        obs = jnp.inf * jnp.ones(self._env.observation_size, dtype=jnp.float32)
        torch_device = jax_to_torch(obs.devices().pop())
        self.single_observation_space: TensorBox = TensorBox(
            -obs,
            obs,
            shape=obs.shape,
            dtype=torch.float32,
            seed=seed,
            device=torch_device,
        )
        self.observation_space: TensorBox = TensorBox(
            -jnp.inf,
            jnp.inf,
            shape=(self.num_envs, *obs.shape),
            dtype=torch.float32,
            seed=seed,
            device=torch_device,
        )
        system: brax.System = self._env.sys
        assert isinstance(system.actuator.ctrl_range, jax.Array)
        # action = jax.tree_map(jnp.array, system.actuator.ctrl_range)  # note: tree_map seems redundant, it's already an array.
        action_range = system.actuator.ctrl_range
        self.single_action_space: TensorBox = TensorBox(
            action_range[:, 0],
            action_range[:, 1],
            dtype=torch.float32,
            seed=seed,
            device=torch_device,
        )
        repeats = tuple([self.num_envs] + [1] * self.single_action_space.low.ndim)
        self.action_space: TensorBox = TensorBox(
            low=jnp.tile(action_range[:, 0], repeats),
            high=jnp.tile(action_range[:, 1], repeats),
            dtype=torch.float32,
            seed=seed,
            device=torch_device,
        )

    def reset_wait(
        self, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict]:
        if isinstance(seed, int):
            self._rng_key = jax.random.key(seed)
        elif seed is not None:
            # todo: maybe it could, but we'd have to bypass the VmapWrapper, which splits a single
            # rng into num_envs..
            raise NotImplementedError("This doesn't work with a list of seeds.")
        self._state, obs, info, self._rng_key = env_reset(self._env, self._rng_key)
        torch_obs = jax_to_torch(obs)
        torch_info = jax_to_torch(info)

        # NOTE: We could try to match the same interface as `SyncVectorEnv` here by adding some dummy
        # boolean masks, but it seems impossible to get the final observation, we'd have to check
        # if the state will be done on this step manually before passing it down to self._env.step,
        # otherwise the AutoReset wrapper from brax will overwrite the pipeline state and the obs.
        bool_mask = torch.ones(self.num_envs, dtype=torch.bool, device=torch_obs.device)
        added_boolean_masks = {f"_{key}": bool_mask for key in torch_info.keys()}
        torch_info = {**torch_info, **added_boolean_masks}
        return torch_obs, torch_info

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        NestedDict[str, torch.Tensor | None],
    ]:
        assert self._state is not None
        jax_action = torch_to_jax(action)

        self._state, obs, reward, terminated, truncated, info = env_step(
            self._env, self._state, jax_action
        )
        torch_obs = jax_to_torch(obs)
        torch_reward = jax_to_torch(reward)
        torch_terminated = jax_to_torch(terminated).bool()
        torch_truncated = jax_to_torch(truncated).bool()
        torch_info = jax_to_torch(info)
        # TODO: Figure out which indices were truncated by inspecting stuff in the EpisodeWrapper.
        # Need to figure out where the first obs of the next episode is, and where the last obs of
        # the current episode is.
        if any(terminated | truncated):
            # Seems way too hard to actually get the final observation and info, it might not
            # actually be worth it...
            pass
        return torch_obs, torch_reward, torch_terminated, torch_truncated, torch_info

    def render(self):
        if self.render_mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("must call reset or step before rendering")
            return brax.io.image.render_array(sys, state.pipeline_state.take(0), 256, 256)
        else:
            return super().render()  # just raise an exception

    # Just in case someone somewhere is actually using this async gym VectorEnv API.
    _actions: torch.Tensor | None = None

    def step_async(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def step_wait(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.BoolTensor,
        torch.BoolTensor,
        NestedDict[str, torch.Tensor | None],
    ]:
        assert self._actions is not None
        return self.step(self._actions)


@functools.partial(jax.jit, static_argnums=(0,))
def env_reset(
    env: brax.envs.Env, key: chex.PRNGKey
) -> tuple[brax.envs.State, jax.Array, NestedDict[str, jax.Array | None], chex.PRNGKey]:
    key1, key2 = jax.random.split(key)
    state = env.reset(key2)
    info = {**state.metrics, **state.info}
    return state, state.obs, info, key1


@functools.partial(jax.jit, static_argnums=(0,))
def env_step(
    env: brax.envs.Env, state: brax.envs.State, action: jax.Array
) -> tuple[
    brax.envs.State,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    NestedDict[str, jax.Array | None],
]:
    state = env.step(state, action)
    # _info_keys = "first_obs", "first_pipeline_state", "steps", "truncation"
    # For HalfCheetah:
    # _metrics_steps = "reward_ctrl", "reward_run", "x_position", "x_velocity"
    truncated = state.info["truncation"].astype(jnp.bool)
    terminated = state.done.astype(jnp.bool)
    # NOTE: reducing the amount of tensors returned. Might not be necessary though.
    # info = {**state.metrics, **state.info}
    info = {**state.metrics, "steps": state.info["steps"]}
    return state, state.obs, state.reward, terminated, truncated, info

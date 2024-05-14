import functools

import gymnasium
import gymnasium.core
import gymnax
import gymnax.environments.spaces
import jax
import torch
from gymnax.wrappers.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

from project.datamodules.rl.rl_types import VectorEnvWrapper
from project.datamodules.rl.wrappers.jax_torch_interop import (
    JaxToTorchMixin,
    get_torch_device_from_jax_array,
    jax_to_torch_tensor,
)
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorSpace,
    get_torch_dtype_from_jax_dtype,
)
from project.utils.device import default_device


def gymnax_env(env_id: str, device: torch.device = default_device(), seed: int = 123):
    # Instantiate the environment & its settings.
    # todo: How to control on which device the brax / gymnax environment resides?
    # For now we just have to assume that it's the same device as `device`, because otherwise we'd
    # be moving stuff between the CPU and GPU!
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToGymWrapper(gymnax_env, params=env_params, seed=seed)
    env = GymnaxToTorchWrapper(env, device=device)
    # NOTE: The spaces are also seeded here.
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


def gymnax_vectorenv(
    env_id: str, num_envs: int = 4096, device: torch.device = default_device(), seed: int = 123
):
    # TODO: Could maybe instead use gymnax.wrappers.brax.GymnaxToBraxWrapper and reuse the same
    # stuff as for Brax. (Would be missing the wrappers from the `create` function of brax though.)

    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    env = GymnaxToVectorGymWrapper(gymnax_env, num_envs=num_envs, params=env_params, seed=seed)
    # Env should already be on the right device (for now).
    # obs = env.reset(seed=123)[0]
    # jax_device = jax.devices("gpu")[0]
    # obs = jax.device_put(obs, )
    # assert get_torch_device_from_jax_array() == device
    return GymnaxVectorEnvToTorchWrapper(env)


class GymnaxToTorchWrapper(
    JaxToTorchMixin, gymnasium.Wrapper[torch.Tensor, torch.Tensor, jax.Array, jax.Array]
):
    def __init__(
        self,
        env: GymnaxToGymWrapper[jax.Array, jax.Array],
        device: torch.device,
    ):
        super().__init__(env=env)
        self.device = device
        self.env: GymnaxToGymWrapper[jax.Array, jax.Array]

        observation_space = self.env._env.observation_space(self.env.env_params)
        action_space = self.env._env.action_space(self.env.env_params)

        self.observation_space = gymnax_space_to_torch_space(observation_space)
        self.action_space = gymnax_space_to_torch_space(action_space)

    def render(self) -> gymnasium.core.RenderFrame | list[gymnasium.core.RenderFrame] | None:
        """use underlying environment rendering if it exists, otherwise return None."""
        return self.env.render()


class GymnaxVectorEnvToTorchWrapper(
    JaxToTorchMixin, VectorEnvWrapper[torch.Tensor, torch.Tensor, jax.Array, jax.Array]
):
    def __init__(self, env: GymnaxToVectorGymWrapper):
        super().__init__(env)  # type: ignore
        self.env: GymnaxToVectorGymWrapper
        jax_single_observation_space = self.env._env.observation_space(self.env.env_params)
        jax_single_action_space = self.env._env.action_space(self.env.env_params)
        torch_single_observation_space = gymnax_space_to_torch_space(jax_single_observation_space)
        torch_single_action_space = gymnax_space_to_torch_space(jax_single_action_space)

        self.single_observation_space = torch_single_observation_space
        self.single_action_space = torch_single_action_space

        # device = torch_single_observation_space.device
        # NOTE: The Gymnax space is in Jax, but they wrap it into a regular Gymnasium space..
        # Here we instead use a TensorSpace that returns torch tensors on the right device.
        self.observation_space = gymnasium.vector.utils.batch_space(
            torch_single_observation_space, self.env.num_envs
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            torch_single_action_space, self.env.num_envs
        )


@functools.singledispatch
def gymnax_space_to_torch_space(
    gymnax_space: gymnax.environments.spaces.Space, /, *, device: torch.device | None = None
) -> TensorSpace:
    raise NotImplementedError(f"No handler for gymnax spaces of type {type(gymnax_space)}")


@gymnax_space_to_torch_space.register
def _gymnax_box_to_torch_box(
    gymnax_space: gymnax.environments.spaces.Box, /, *, device: torch.device | None = None
) -> TensorBox:
    if not device:
        jax_array = gymnax_space.sample(jax.random.key(0))
        assert isinstance(jax_array, jax.Array)
        device = get_torch_device_from_jax_array(jax_array)

    def _keep_as_float_or_to_tensor(v: float | jax.Array):
        if isinstance(v, jax.Array):
            return jax_to_torch_tensor(v)
        return v

    return TensorBox(
        low=_keep_as_float_or_to_tensor(gymnax_space.low),
        high=_keep_as_float_or_to_tensor(gymnax_space.high),
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

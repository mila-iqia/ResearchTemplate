from collections.abc import Sequence
from typing import Any, SupportsFloat, TypedDict

import gymnasium
import gymnasium.envs.registration
import numpy as np
import torch

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorMultiDiscrete,
)


class DebugEnvInfo(TypedDict):
    episode_length: torch.Tensor
    target: torch.Tensor


class DebugEnv(gymnasium.Env[torch.Tensor, torch.Tensor]):
    """A simple environment for debugging.

    The goal is to match the state with a hidden target state by adding 1, subtracting 1, or
    staying the same.
    """

    def __init__(
        self,
        min: int = -10,
        max: int = 10,
        target: int = 5,
        initial_state: int = 0,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        wrap_around_state: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.int32,
        seed: int | None = None,
    ):
        # Don't call super().__init__ because it would cause an error below.
        # super().__init__()

        self.min = min
        self.max = max
        self.max_episode_length = max_episode_length

        self.randomize_target = randomize_target
        self.randomize_initial_state = randomize_initial_state

        self.wrap_around_state = wrap_around_state
        self.device = device
        self.dtype = dtype
        self.observation_space: TensorBox = TensorBox(
            low=self.min, high=self.max, shape=(), dtype=self.dtype, device=self.device
        )
        # todo: make this a TensorBox(-1, 1) for a version with a continuous action space.
        self.action_space: TensorDiscrete = TensorDiscrete(
            start=-1, n=3, dtype=self.dtype, device=self.device
        )
        self._episode_length = torch.zeros(
            self.observation_space.shape, dtype=torch.int32, device=device
        )

        self._target = torch.as_tensor(target, dtype=dtype, device=device)
        assert self._target in self.observation_space, "invalid target!"

        self._initial_state = torch.as_tensor(initial_state, dtype=dtype, device=device)
        assert self._initial_state in self.observation_space, "invalid initial state!"
        self._state = self._initial_state.clone()

        self.spec = gymnasium.envs.registration.EnvSpec(
            id="DebugEnv-v0",
            entry_point="project.datamodules.rl.envs.debug_env:DebugEnv",
            max_episode_steps=max_episode_length,
            vector_entry_point="project.datamodules.rl.envs.debug_env:DebugVectorEnv",
        )
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, DebugEnvInfo]:
        if seed:
            self.rng.manual_seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        if self.randomize_target:
            # Set a new target for the next episode.
            self._target = self.observation_space.sample()

        if self.randomize_initial_state:
            # Set a new initial state for the next episode.
            self._state = self.observation_space.sample()
        else:
            self._state = self._initial_state.clone()

        self._episode_length = torch.zeros_like(self._episode_length)
        return (
            self._state,
            {"episode_length": self._episode_length, "target": self._target},
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, SupportsFloat, torch.BoolTensor, torch.BoolTensor, DebugEnvInfo]:
        if action not in self.action_space:
            raise RuntimeError(
                f"Invalid action: {action} of type {type(action)} , {action.dtype=}, {action.device=} "
                f"is not in {self.action_space}."
                + (
                    f" (wrong device: {action.device}!={self.action_space.device})"
                    if action.device != self.action_space.device
                    else ""
                )
                + (
                    f" (wrong dtype: {action.dtype} can't be casted to {self.action_space.dtype})"
                    if not torch.can_cast(action.dtype, self.action_space.dtype)
                    else ""
                )
            )
        assert torch.can_cast(action.dtype, self.action_space.dtype), (
            action.dtype,
            self.action_space.dtype,
        )

        self._state += action
        # Two options: Either we wrap around, or we clamp to the range.
        if self.wrap_around_state:
            # -11 -> 9
            self._state = torch.where(
                self._state < self.min, self.max + (self.min - self._state), self._state
            )
            # 11 -> -9
            self._state = torch.where(
                self._state > self.max, self.min + (self._state - self.max), self._state
            )
        else:
            self._state = torch.clamp(
                self._state,
                min=torch.zeros_like(self._state),
                max=self.max * torch.ones_like(self._state),
            )
        assert self._target is not None
        reward = -(self._state - self._target).abs().to(dtype=torch.float32)
        self._episode_length += 1
        episode_ended = self._episode_length == self.max_episode_length
        at_target = self._state == self._target
        terminated: torch.BoolTensor = episode_ended & at_target  # type: ignore
        truncated: torch.BoolTensor = episode_ended & ~at_target  # type: ignore
        return (
            self._state.clone(),
            reward,
            terminated,
            truncated,
            {"episode_length": self._episode_length.clone(), "target": self._target.clone()},
        )


class DebugVectorEnvInfo(TypedDict):
    episode_length: torch.Tensor
    target: torch.Tensor
    final_observation: np.ndarray | Sequence[torch.Tensor | None]
    _final_observation: torch.BoolTensor
    final_info: np.ndarray | Sequence[DebugEnvInfo | None]
    _final_info: torch.BoolTensor


class DebugVectorEnv(DebugEnv, VectorEnv[torch.Tensor, torch.Tensor]):
    """Same as DebugEnv, but vectorized."""

    def __init__(
        self,
        num_envs: int,
        min: int = -10,
        max: int = 10,
        target: int = 5,
        initial_state: int = 0,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        wrap_around_state: bool = False,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.int32,
    ):
        self.num_envs = num_envs
        super().__init__(
            min=min,
            max=max,
            target=target,
            initial_state=initial_state,
            max_episode_length=max_episode_length,
            randomize_target=randomize_target,
            randomize_initial_state=randomize_initial_state,
            wrap_around_state=wrap_around_state,
            device=device,
            seed=seed,
            dtype=dtype,
        )
        single_observation_space = self.observation_space
        single_action_space = self.action_space
        # todo: double-check that 1 is in this this space (bounds are included).
        VectorEnv.__init__(
            self,
            num_envs=num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )
        expected_observation_space = TensorBox(
            low=self.min,
            high=self.max,
            shape=(self.num_envs,),
            dtype=self.dtype,
            device=self.device,
        )
        assert self.observation_space == expected_observation_space, (
            expected_observation_space,
            self.observation_space,
        )
        expected_action_space = TensorMultiDiscrete(
            start=torch.full(
                (self.num_envs,), fill_value=-1, dtype=self.dtype, device=self.device
            ),
            nvec=torch.full((self.num_envs,), fill_value=3, dtype=self.dtype, device=self.device),
            dtype=self.dtype,
            device=self.device,
        )
        assert self.action_space == expected_action_space, (
            expected_action_space,
            self.action_space,
        )
        if seed is not None:
            self.single_observation_space.seed(seed)
            self.single_action_space.seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self._episode_length = self._episode_length.expand((self.num_envs,))
        self._state = self._state.expand(self.observation_space.shape)
        self._target = self._target.expand(self.observation_space.shape)
        self._initial_state = self._initial_state.expand(self.observation_space.shape)

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        SupportsFloat,
        torch.BoolTensor,
        torch.BoolTensor,
        DebugVectorEnvInfo | DebugEnvInfo,
    ]:
        obs, reward, terminated, truncated, info = super().step(action)
        env_done: torch.BoolTensor = terminated | truncated  # type: ignore

        if env_done.any():
            old_observation = obs
            if self.randomize_initial_state:
                self._state[env_done] = self.observation_space.sample()[env_done]
            else:
                self._state[env_done] = self._initial_state.clone()[env_done]
            obs = self._state.clone()

            old_target = self._target.clone()
            if self.randomize_target:
                # Set a new target for the next episode.
                self._target[env_done] = self.observation_space.sample()[env_done]

            old_episode_length = self._episode_length.clone()
            self._episode_length[env_done] = 0

            old_info: DebugEnvInfo = {
                "episode_length": torch.where(env_done, old_episode_length, self._episode_length),
                "target": torch.where(env_done, old_target, self._target),
            }
            # We have to try to match gymnasium.vector.VectorEnv, so the final observations should be a
            # list (or numpy array of objects).
            info: DebugVectorEnvInfo = {
                "episode_length": self._episode_length.clone(),
                "target": self._target.clone(),
                # NOTE: We're not actually able to use a np.ndarray here to perfectly match the
                # VectorEnv, because it would try to convert the cuda tensors to numpy arrays.
                # We'll just keep this as a list for now.
                "final_observation": [
                    old_observation_i if env_done[i] else None
                    for i, old_observation_i in enumerate(old_observation)
                ],
                # dtype=object,
                # copy=False,
                # Todo: Look at the `like` argument of `np.asarray`, could be very interesting
                # to start using it in gym so the ndarrays created can actually be jax Arrays
                # ),
                "_final_observation": env_done,
                "final_info": np.array(
                    [
                        {k: v[i] for k, v in old_info.items()} if env_done_i else None
                        for i, env_done_i in enumerate(env_done)
                    ],
                    dtype=object,
                ),
                "_final_info": env_done,
            }
        return obs, reward, terminated, truncated, info

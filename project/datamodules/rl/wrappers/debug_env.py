from typing import Any, SupportsFloat

import gymnasium
import numpy as np
import torch

from project.datamodules.rl.rl_types import VectorEnv
from project.datamodules.rl.wrappers.tensor_spaces import TensorBox, TensorDiscrete


class DebugEnv(gymnasium.Env[torch.Tensor, torch.Tensor]):
    """A simple environment for debugging.

    The goal is to match the state with a hidden target state by adding 1, subtracting 1, or
    staying the same.
    """

    def __init__(
        self,
        max: int = 10,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype=torch.int32,
    ):
        super().__init__()
        self.max = max
        self.max_episode_length = max_episode_length
        self.randomize_target = randomize_target
        self.randomize_initial_state = randomize_initial_state
        self.device = device
        self.dtype = dtype
        self.rng = torch.Generator(device=self.device)
        # todo: make this a TensorBox(-1, 1) for a version with a continuous action space.
        self.action_space = TensorDiscrete(n=3, start=-1, dtype=self.dtype, device=self.device)
        self.observation_space: TensorDiscrete = TensorDiscrete(
            n=self.max + 1, start=0, dtype=self.dtype, device=self.device
        )
        self._episode_length = np.zeros(self.observation_space.shape, dtype=int)
        self.reset(seed=seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed:
            self.rng.manual_seed(seed)

        if self.randomize_target:
            self._target = torch.randint(
                0,
                self.max,
                size=self.observation_space.shape,
                dtype=self.observation_space.dtype,
                device=self.device,
                generator=self.rng,
            )
        else:
            self._target = (self.max // 2) * torch.ones(
                size=self.observation_space.shape,
                dtype=self.observation_space.dtype,
                device=self.device,
            )

        if self.randomize_initial_state:
            self._state = torch.randint(
                0,
                self.max,
                size=self._target.shape,
                dtype=torch.int32,
                device=self.device,
                generator=self.rng,
            )
        else:
            self._state = torch.zeros_like(self._target)

        self._episode_length.fill(0)
        return (
            self._state,
            {"episode_length": self._episode_length, "target": self._target},
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, SupportsFloat, bool, bool, dict[str, Any]]:
        if action not in self.action_space:
            raise RuntimeError(f"Invalid action: {action} not in {self.action_space}.")
        assert action.dtype == self.action_space.dtype

        self._state += action
        reward = -(self._state - self._target).abs()
        self._episode_length += 1
        done = self._episode_length == self.max_episode_length
        truncated = done & (self._state != self._target)
        return (
            self._state,
            reward,
            done,
            truncated,
            {"episode_length": self._episode_length, "target": self._target},
        )


class DebugVectorEnv(DebugEnv, VectorEnv[torch.Tensor, torch.Tensor]):
    """Same as DebugEnv, but vectorized."""

    def __init__(
        self,
        num_envs: int,
        max: int = 10,
        max_episode_length: int = 20,
        randomize_target: bool = False,
        randomize_initial_state: bool = False,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.int32,
    ):
        self.num_envs = num_envs
        super().__init__(
            max=max,
            max_episode_length=max_episode_length,
            randomize_target=randomize_target,
            randomize_initial_state=randomize_initial_state,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        # TODO: would be nice to do this, but it would require us to implement something like
        # MultiDiscrete in torch and register a handler for `TensorDiscrete` on `batch_space` that
        # would return it.
        # self.observation_space: gymnasium.Space[torch.Tensor] = gymnasium.vector.utils.batch_space(
        #     self.observation_space, num_envs
        # )
        # self.action_space: gymnasium.Space[torch.Tensor] = gymnasium.vector.utils.batch_space(
        #     self.action_space, num_envs
        # )
        self.observation_space: TensorBox = TensorBox(
            0, high=self.max, shape=(self.num_envs,), dtype=self.dtype, device=self.device
        )
        # todo: double-check that 1 is in this this space (bounds are included).
        self.action_space: TensorBox = TensorBox(
            -1, high=1, shape=(self.num_envs,), dtype=self.dtype, device=self.device
        )
        self._episode_length = np.zeros(shape=(num_envs,), dtype=int)

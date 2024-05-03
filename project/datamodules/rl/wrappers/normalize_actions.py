from typing import Any

import numpy as np
from gymnasium.core import ActionWrapper

from ..rl_types import BoxSpace, _Env


class NormalizeBoxActionWrapper(ActionWrapper):
    """Wrapper to normalize gym.spaces.Box actions in [-1, 1].

    TAKEN FROM (https://github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py)
    """

    def __init__(self, env: _Env[Any, np.ndarray]):
        if not isinstance(env.action_space, BoxSpace):
            raise ValueError(f"env {env} doesn't have a Box action space.")
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.action_space = type(env.action_space)(
            low=np.ones_like(env.action_space.low) * -1.0,
            high=np.ones_like(env.action_space.high),
            dtype=env.action_space.dtype,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        # rescale the action
        low, high = self.orig_action_space.low, self.orig_action_space.high
        scaled_action = low + (action + 1.0) * (high - low) / 2.0
        scaled_action = np.clip(scaled_action, low, high)
        return scaled_action

    def reverse_action(self, scaled_action: np.ndarray) -> np.ndarray:
        low, high = self.orig_action_space.low, self.orig_action_space.high
        action = (scaled_action - low) * 2.0 / (high - low) - 1.0
        return action

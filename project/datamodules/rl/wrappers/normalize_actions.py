from logging import getLogger as get_logger
from typing import Any

import gymnasium
import numpy as np
import torch

from project.datamodules.rl.wrappers.tensor_spaces import TensorBox

from ..types import BoxSpace

logger = get_logger(__name__)


def _ones_like[T: np.ndarray | torch.Tensor](v: T) -> T:
    return np.ones_like(v) if isinstance(v, np.ndarray) else torch.ones_like(v)


class NormalizeBoxActionWrapper[ObsType, ActionType: np.ndarray | torch.Tensor](
    gymnasium.ActionWrapper[ObsType, ActionType, ActionType]
):
    """Wrapper to normalize gym.spaces.Box actions in [-1, 1].

    Adapted from (https://github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py)
    """

    def __init__(self, env: gymnasium.Env[Any, ActionType]):
        if not isinstance(env.action_space, BoxSpace | TensorBox):
            raise ValueError(f"env {env} doesn't have a Box action space.")
        super().__init__(env)
        self.orig_action_space = env.action_space
        if isinstance(env.action_space, BoxSpace):
            action_space = gymnasium.spaces.Box(
                low=_ones_like(env.action_space.low) * -1.0,
                high=_ones_like(env.action_space.high),
                dtype=env.action_space.dtype,
            )
        else:
            action_space = TensorBox(
                low=-_ones_like(env.action_space.low) * -1.0,
                high=_ones_like(env.action_space.high),
                dtype=env.action_space.dtype,
                device=env.action_space.device,
            )
        self.action_space: BoxSpace | TensorBox = action_space

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


def check_and_normalize_box_actions(env: gymnasium.Env):
    """Wrap env to normalize actions if [low, high] != [-1, 1]."""
    if isinstance(env.action_space, BoxSpace | TensorBox):
        low, high = env.action_space.low, env.action_space.high
        if (low != -1).any() or (high != 1).any():
            logger.info("Normalizing environment actions.")
            return NormalizeBoxActionWrapper(env)
    # Environment does not need to be normalized.
    return env

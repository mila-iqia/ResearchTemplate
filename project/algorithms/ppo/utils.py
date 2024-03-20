from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from torch import Tensor

from project.datamodules.rl.rl_datamodule import EpisodeBatch


class PPOActorOutput(TypedDict):
    """Additional outputs of the Actor (besides the actions to take) at a given step in the env.

    This should be used to store whatever is needed to train the model later (e.g. the action log-
    probabilities, activations, etc.)
    """

    action_mean: Tensor
    """The network outputs (action mean) at that step."""

    action_std: Tensor
    """The action standard deviation at that step."""

    action_log_probability: Tensor
    """The log-probability of the selected actions at each step."""

    values: Tensor
    """Value function estimates for each observation."""


PpoEpisodeBatch = EpisodeBatch[PPOActorOutput]


def discount_cumsum(x: Tensor, discount: float, dim: int = -1) -> Tensor:
    """Discounted cumulative sums of vectors.

    >>> import torch
    >>> discount_cumsum(torch.ones(5), 0.5)
    tensor([1.9375, 1.8750, 1.7500, 1.5000, 1.0000])
    >>> discount_cumsum(torch.arange(3)+1, 1.0)
    tensor([6., 5., 3.])
    >>> discount_cumsum(torch.ones(8), 0.99)
    tensor([7.7255, 6.7935, 5.8520, 4.9010, 3.9404, 2.9701, 1.9900, 1.0000])
    """
    # import scipy.signal

    # return scipy.signal.lfilter([1], [1, float(-discount)], x.cpu().numpy()[::-1], axis=0)[::-1]
    if x.is_nested:
        return torch.nested.as_nested_tensor(
            [
                discount_cumsum(x_i, discount=discount, dim=-1 if dim == -1 else dim - 1)
                for x_i in x.unbind()
            ]
        )
    # TODO: Find a neat way of doing this without for-loops using only pytorch ops.
    returns = torch.zeros_like(x)
    rewards_at_each_step = x.unbind(dim=dim)
    discounted_future_values = torch.zeros_like(rewards_at_each_step[0])
    returns_list = []
    for reward_at_that_step in reversed(rewards_at_each_step):
        discounted_future_values = reward_at_that_step + discount * discounted_future_values
        # TODO: This is perhaps slower, but doesn't assume that dim=0
        returns_list.append(discounted_future_values)
        # returns[step] = discounted_future_values

    returns = torch.stack(returns_list[::-1], dim=dim)
    return returns

    # seq_length = x.size(dim)
    # exponents = torch.ones((seq_length, seq_length), device=x.device).triu_(diagonal=1)

    # # x = torch.arange(9).view(3, 3)

    # discounts = torch.pow(discount, torch.arange(x.size(dim), device=x.device))
    # weighted_x = x * discounts
    # # Reversed Cumsum without a flip trick from https://github.com/pytorch/pytorch/issues/33520#issuecomment-812907290
    # # cumsum = torch.cumsum(weighted, dim=dim).flip(dim=dim)
    # cumsum = (
    #     weighted_x + weighted_x.sum(dim=dim, keepdim=True) - torch.cumsum(weighted_x, dim=dim)
    # )
    # return cumsum


def stack_or_raise(nested_tensor: Tensor) -> Tensor:
    if not nested_tensor.is_nested:
        return nested_tensor
    if not all(out_i.shape == nested_tensor[0].shape for out_i in nested_tensor.unbind()):
        raise NotImplementedError(
            "Can't pass a nested tensor to all ops yet. Therefore we temporarily assume here that "
            "we have the same shape for all items in the nested tensor."
        )
    return torch.stack(nested_tensor.unbind())


def get_episode_lengths(episodes: EpisodeBatch) -> list[int]:
    if episodes["rewards"].is_nested:
        return [rewards_i.size(0) for rewards_i in episodes["rewards"].unbind()]
    return [episodes["rewards"].size(1) for _ in range(episodes["rewards"].size(0))]


def make_dense_if_possible(possibly_nested_tensor: Tensor) -> tuple[Tensor, bool]:
    """Turns a nested tensor into a dense tensor if all shapes are the same in the sub-tensors.

    Returns the resulting tensors and a boolean indicating whether this was successful (and the
    resulting tensor is dense).
    """
    if not possibly_nested_tensor.is_nested:
        return possibly_nested_tensor, True
    if all(
        v_i.shape == possibly_nested_tensor[0].shape for v_i in possibly_nested_tensor.unbind()
    ):
        # They are all the same shape, so it's fine to "pad" the nested tensor, since there won't
        # be any padding necessary. (I'm thinking that this might be faster than doing unbind() +
        # stack, but I don't know for sure.)
        return torch.nested.to_padded_tensor(possibly_nested_tensor, 0), True
    return possibly_nested_tensor, False


def concatenate_episode_batches(dicts: list[PpoEpisodeBatch], dim=0) -> PpoEpisodeBatch:
    def _concat(tensors: list[Tensor], dim: int = 0) -> Tensor:
        if tensors[0].is_nested:
            assert dim == 0, "only support concatenating with dim=0 when the tensors are nested"
            return torch.nested.as_nested_tensor(sum((t_i.unbind() for t_i in tensors), []))
        return torch.concatenate(tensors, dim=dim)

    return EpisodeBatch(
        observations=_concat([d["observations"] for d in dicts], dim=dim),
        actions=_concat([d["actions"] for d in dicts], dim=dim),
        rewards=_concat([d["rewards"] for d in dicts], dim=dim),
        infos=sum((d["infos"] for d in dicts), []),
        truncated=torch.concatenate([d["truncated"] for d in dicts], dim=dim),
        terminated=torch.concatenate([d["terminated"] for d in dicts], dim=dim),
        actor_outputs=type(dicts[0]["actor_outputs"])(
            **{
                k: _concat([d["actor_outputs"][k] for d in dicts], dim=dim)
                for k in dicts[0]["actor_outputs"].keys()
            }
        ),
    )


def dict_slice(d: PpoEpisodeBatch, indices: np.ndarray) -> PpoEpisodeBatch:
    torch_indices = torch.as_tensor(indices, device=d["observations"].device)
    return EpisodeBatch(
        observations=d["observations"][torch_indices],
        actions=d["actions"][torch_indices],
        rewards=d["rewards"][torch_indices],
        infos=[d["infos"][i] for i in indices],
        actor_outputs=type(d["actor_outputs"])(
            {k: v[torch_indices] for k, v in d["actor_outputs"].items()}
        ),
    )

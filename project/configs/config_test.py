"""TODO: tests for the configs?"""

import functools

import hydra_zen
import pytest
import torch
from hydra_zen.third_party.pydantic import pydantic_parser
from hydra_zen.typing import PartialBuilds

from project.configs.algorithm.lr_scheduler import get_all_scheduler_configs
from project.configs.algorithm.optimizer import get_all_optimizer_configs
from project.utils.testutils import seeded


@pytest.fixture(scope="session")
def net(device: torch.device):
    with seeded(123):
        net = torch.nn.Linear(10, 1).to(device)
    return net


@pytest.mark.parametrize("optimizer_config", get_all_optimizer_configs())
def test_optimizer_configs(
    optimizer_config: type[PartialBuilds[torch.optim.Optimizer]], net: torch.nn.Module
):
    assert hydra_zen.is_partial_builds(optimizer_config)
    target = hydra_zen.get_target(optimizer_config)
    assert issubclass(target, torch.optim.Optimizer)

    optimizer_partial = hydra_zen.instantiate(optimizer_config)
    assert isinstance(optimizer_partial, functools.partial)

    optimizer = optimizer_partial(net.parameters())

    assert isinstance(optimizer, torch.optim.Optimizer), optimizer
    assert isinstance(optimizer, target)


# This could also be used to test with all optimizers, but isn't necessary.
# @pytest.fixture(scope="session", params=get_all_optimizer_configs())
# @pytest.fixture(scope="session")
# def optimizer(device: torch.device, net: torch.nn.Module):
#     # optimizer_config: type[PartialBuilds[torch.optim.Optimizer]] = request.param
#     # optimizer = hydra_zen.instantiate(optimizer_config)(net.parameters())
#     return torch.optim.SGD(net.parameters(), lr=0.1)
#     return optimizer


_optim = torch.optim.SGD([torch.zeros(1, requires_grad=True)])

default_kwargs: dict[type[torch.optim.lr_scheduler.LRScheduler], dict] = {
    torch.optim.lr_scheduler.StepLR: {"step_size": 1},
    torch.optim.lr_scheduler.CosineAnnealingLR: {"T_max": 10},
    torch.optim.lr_scheduler.LambdaLR: {"lr_lambda": lambda epoch: 0.95**epoch},
    torch.optim.lr_scheduler.MultiplicativeLR: {"lr_lambda": lambda epoch: 0.95**epoch},
    torch.optim.lr_scheduler.MultiStepLR: {"milestones": [0, 1]},
    torch.optim.lr_scheduler.ExponentialLR: {"gamma": 0.8},
    torch.optim.lr_scheduler.SequentialLR: {
        "schedulers": [
            torch.optim.lr_scheduler.ExponentialLR(_optim, gamma=0.9),
            torch.optim.lr_scheduler.ExponentialLR(_optim, gamma=0.9),
        ],
        "milestones": [0],
    },
    torch.optim.lr_scheduler.CyclicLR: {"base_lr": 0.1, "max_lr": 0.9},
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: {"T_0": 1},
    torch.optim.lr_scheduler.OneCycleLR: {"max_lr": 1.0, "total_steps": 10},
}
"""The missing arguments for some LR schedulers so we can create them during testing.

The values don't really matter, as long as they are accepted by the constructor.
"""


schedulers_to_skip = {
    # torch.optim.lr_scheduler.SequentialLR: "Requires other schedulers as arguments. Ugly."
    torch.optim.lr_scheduler.ChainedScheduler: "Requires passing a list of schedulers as arguments."
}


# pytest.param(scheduler_config, marks=[pytest.mark.skipif(hydra_zen.get_target(scheduler_config) is torch.optim.lr_scheduler.SequentialLR, reason="Requires other schedulers as arguments. Ugly.")
@pytest.mark.parametrize("scheduler_config", get_all_scheduler_configs())
def test_scheduler_configs(
    scheduler_config: type[PartialBuilds[torch.optim.Optimizer]],
    net: torch.nn.Module,
    # optimizer: torch.optim.Optimizer,
):
    assert hydra_zen.is_partial_builds(scheduler_config)
    target = hydra_zen.get_target(scheduler_config)
    if target in schedulers_to_skip:
        pytest.skip(reason=schedulers_to_skip[target])

    assert issubclass(target, torch.optim.lr_scheduler.LRScheduler)

    scheduler_partial = hydra_zen.instantiate(
        scheduler_config, _target_wrapper_=pydantic_parser, **default_kwargs.get(target, {})
    )
    assert isinstance(scheduler_partial, functools.partial)

    lr_scheduler = scheduler_partial(_optim)
    assert isinstance(lr_scheduler, target)

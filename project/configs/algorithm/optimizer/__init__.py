import functools
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch
from hydra_zen import hydrated_dataclass, instantiate

# OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer, covariant=True)


@runtime_checkable
class OptimizerConfig[OptimizerType: torch.optim.Optimizer](Protocol):
    """Configuration for an optimizer.

    Returns a partial[OptimizerType] when instantiated.
    """

    __call__: Callable[..., functools.partial[OptimizerType]] = instantiate


# NOTE: Getting weird bug in omegaconf if I try to make OptimizerConfig generic!
# Instead I'm making it a protocol.


@hydrated_dataclass(target=torch.optim.SGD, zen_partial=True)
class SGDConfig:
    """Configuration for the SGD optimizer."""

    lr: Any
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    __call__ = instantiate


# TODO: frozen=True doesn't work here, which is a real bummer. It would have saved us a lot of
# functools.partial(AdamConfig, lr=123) nonsense.
@hydrated_dataclass(target=torch.optim.Adam, zen_partial=True)
class AdamConfig:
    """Configuration for the Adam optimizer."""

    lr: Any = 0.001
    betas: Any = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False

    __call__ = instantiate


# NOTE: we don't add an `optimizer` group, since models could have one or more optimizers.
# Models can register their own groups, e.g. `model/optimizer`. if they want to.
# cs = ConfigStore.instance()
# cs.store(group="optimizer", name="sgd", node=SGDConfig)
# cs.store(group="optimizer", name="adam", node=AdamConfig)

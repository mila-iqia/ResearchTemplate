from typing import TypeVar

import torch.optim.lr_scheduler
from hydra.core.config_store import ConfigStore
from hydra_zen import hydrated_dataclass
from hydra_zen.typing._implementations import PartialBuilds
from typing_extensions import TypeAlias

LRSchedulerType = TypeVar("LRSchedulerType", bound=torch.optim.lr_scheduler._LRScheduler)
# TODO: Double-check this, but defining LRSchedulerConfig like this makes it unusable as a type
# annotation on the hparams, since omegaconf will complain it isn't a valid base class.
LRSchedulerConfig: TypeAlias = PartialBuilds[LRSchedulerType]


@hydrated_dataclass(target=torch.optim.lr_scheduler.StepLR, zen_partial=True)
class StepLRConfig:
    """Config for the StepLR Scheduler."""

    step_size: int = 30
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@hydrated_dataclass(target=torch.optim.lr_scheduler.CosineAnnealingLR, zen_partial=True)
class CosineAnnealingLRConfig:
    """Config for the CosineAnnealingLR Scheduler."""

    T_max: int = 85
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


# cs = ConfigStore.instance()
# cs.store(group="lr_scheduler", name="step", node=StepLRConfig)
# cs.store(group="lr_scheduler", name="cosine", node=CosineAnnealingLRConfig)

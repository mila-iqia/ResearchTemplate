import torch
import torch.optim.lr_scheduler
from hydra_zen import make_custom_builds_fn, store
from hydra_zen.third_party.pydantic import pydantic_parser

builds_fn = make_custom_builds_fn(
    zen_partial=True, populate_full_signature=True, zen_wrappers=pydantic_parser
)

CosineAnnealingLRConfig = builds_fn(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=85)
StepLRConfig = builds_fn(torch.optim.lr_scheduler.CosineAnnealingLR)
lr_scheduler_store = store(group="algorithm/lr_scheduler")
lr_scheduler_store(StepLRConfig, name="step_lr")
lr_scheduler_store(CosineAnnealingLRConfig, name="cosine_annealing_lr")


# IDEA: Could be interesting to generate configs for any member of the torch.optimizer.lr_scheduler
# package dynamically (and store it)?
# def __getattr__(self, name: str):
#     """"""

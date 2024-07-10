import torch
import torch.optim
from hydra_zen import make_custom_builds_fn, store
from hydra_zen.third_party.pydantic import pydantic_parser

builds_fn = make_custom_builds_fn(
    zen_partial=True, populate_full_signature=True, zen_wrappers=pydantic_parser
)

optimizer_store = store(group="algorithm/optimizer")
AdamConfig = builds_fn(torch.optim.Adam)
SGDConfig = builds_fn(torch.optim.SGD)
optimizer_store(AdamConfig, name="adam")
optimizer_store(SGDConfig, name="sgd")

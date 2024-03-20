from typing import Any, Required, TypedDict

from torch import Tensor


class StepOutputDict(TypedDict, total=False):
    """A dictionary that shows what an Algorithm should output from `training/val/test_step`."""

    loss: Tensor | float
    """Optional loss tensor that can be returned by those methods."""

    log: dict[str, Tensor | Any]
    """Optional dictionary of things to log at each step."""
    # TODO: Remove this `log` dict, perhaps it's better to use self.log of the pl module instead?


class ClassificationOutputs(StepOutputDict):
    """The dictionary format that is minimally required to be returned from
    `training/val/test_step`."""

    logits: Required[Tensor]
    """The un-normalized logits."""

    y: Required[Tensor]
    """The class labels."""

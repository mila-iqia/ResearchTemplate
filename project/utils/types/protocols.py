from __future__ import annotations

import dataclasses
import typing
from collections.abc import Iterable
from typing import ClassVar, Protocol, runtime_checkable

from torch import nn

if typing.TYPE_CHECKING:
    from project.utils.types import StageStr


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


@runtime_checkable
class Module[**P, OutT](Protocol):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> OutT:
            ...

        modules = nn.Module.modules
        named_modules = nn.Module.named_modules
        state_dict = nn.Module.state_dict
        zero_grad = nn.Module.zero_grad
        parameters = nn.Module.parameters
        named_parameters = nn.Module.named_parameters
        cuda = nn.Module.cuda
        cpu = nn.Module.cpu
        # note: the overloads on nn.Module.to cause a bug with missing `self`.
        # This shouldn't be a problem.
        to = nn.Module().to


@runtime_checkable
class HasInputOutputShapes(Module, Protocol):
    """Protocol for a a module that is "easy to invert" since it has known input and output shapes.

    It's easier to mark modules as invertible in-place than to create new subclass for every single
    nn.Module class that we want to potentially use in the forward net.
    """

    input_shape: tuple[int, ...]
    # input_shapes: tuple[tuple[int, ...] | None, ...] = ()
    output_shape: tuple[int, ...]


@runtime_checkable
class DataModule[BatchType](Protocol):
    """Protocol that shows the expected attributes / methods of the `LightningDataModule` class.

    This is used to type hint the batches that are yielded by the DataLoaders.
    """

    # batch_size: int

    def prepare_data(self) -> None:
        ...

    def setup(self, stage: StageStr) -> None:
        ...

    def train_dataloader(self) -> Iterable[BatchType]:
        ...

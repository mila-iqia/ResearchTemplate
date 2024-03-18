import typing
from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class Module[**P, OutT](Protocol):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> OutT: ...

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

from __future__ import annotations

import functools
import operator
import typing
from logging import getLogger as get_logger
from typing import Any, Callable, Union

import torch
from torch import Tensor, nn

from project.utils.types import Module, OutT, P, T, is_sequence_of

logger = get_logger(__name__)


class Lambda(nn.Module, Module[..., OutT]):
    """A simple nn.Module wrapping a function.

    Any positional or keyword arguments passed to the constructor are saved into a `args` and
    `kwargs` attribute. During the forward pass, these arguments are then bound to the function `f`
    using a `functools.partial`. Any additional arguments to the `forward` method are then passed
    to the partial.
    """

    __match_args__ = ("f",)

    def __init__(
        self,
        f: Callable[..., OutT],
        *args: Tensor | nn.Module | Any,
        **kwargs: Tensor | nn.Module | Any,
    ):
        super().__init__()
        self.f: Callable[..., OutT] = f
        self.args: nn.ParameterList | nn.ModuleList | tuple[Any, ...]
        if not args:
            self.args = args
        elif is_sequence_of(args, Tensor):
            self.args = nn.ParameterList(
                arg
                if isinstance(arg, nn.Parameter)
                else nn.Parameter(arg, requires_grad=arg.requires_grad)
                for arg in args
            )
        elif is_sequence_of(args, nn.Module):
            self.args = nn.ModuleList(args)
        else:
            self.args = args
            # raise NotImplementedError(f"Args need to either be tensors or modules, not {args}")
        self.kwargs: nn.ParameterDict | nn.ModuleDict | dict[str, Any]
        if not kwargs:
            self.kwargs = kwargs
        elif is_sequence_of(kwargs.values(), Tensor):
            self.kwargs = nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad=v.requires_grad)
                    for k, v in kwargs.items()
                }  # type: ignore
            )
        elif is_sequence_of(kwargs.values(), nn.Module):
            self.kwargs = nn.ModuleDict(kwargs)  # type: ignore
        else:
            self.kwargs = kwargs
            # raise NotImplementedError(f"kwargs need to either be tensors or modules, not {kwargs}")

    def forward(self, *args, **kwargs) -> OutT:
        f = functools.partial(self.f, *self.args, **self.kwargs)
        return f(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in ["f", "args", "kwargs"]:
            return super().__getattr__(name)
        if name in (kwargs := self.kwargs):
            logger.debug(f"Getting {name} from kwargs: {kwargs}")
            return kwargs[name]
        return super().__getattr__(name)

    def extra_repr(self) -> str:
        # TODO: Redo, now that we have `args` and `kwargs` in containers.
        func_message: str = ""

        if isinstance(self.f, nn.Module):
            func_message = ""
        elif isinstance(self.f, functools.partial):
            assert not self.args and not self.kwargs
            partial_args: list[str] = []
            if not isinstance(self.f.func, nn.Module):
                partial_args.append(f"{self.f.func.__module__}.{self.f.func.__name__}")
            else:
                partial_args.append(repr(self.f.func))
            partial_args.extend(repr(x) for x in self.f.args)
            partial_args.extend(f"{k}={v!r}" for (k, v) in self.f.keywords.items())

            partial_qualname = type(self.f).__qualname__
            if type(self.f).__module__ == "functools":
                partial_qualname = f"functools.{partial_qualname}"
            func_message = f"f={partial_qualname}(" + ", ".join(partial_args) + ")"
        elif hasattr(self.f, "__module__") and hasattr(self.f, "__name__"):
            module = self.f.__module__
            if module != "builtins":
                module += "."
            func_message = f"f={module}{self.f.__name__}"
        else:
            func_message = f"f={self.f}"

        args_message = ""
        if isinstance(self.args, (nn.ParameterList, nn.ModuleList)):
            args_message = ""
        elif self.args:
            args_message = ", ".join(f"{arg!r}" for (arg) in self.args)

        kwargs_message = ""
        if isinstance(self.kwargs, (nn.ParameterDict, nn.ModuleDict)):
            kwargs_message = ""
        elif self.kwargs:
            kwargs_message = ", ".join(f"{k}={v!r}" for (k, v) in self.kwargs.items())

        message = ""
        if func_message:
            message += func_message
        if args_message:
            message += (", " if message else "") + args_message
        if kwargs_message:
            message += (", " if message else "") + kwargs_message
        return message

    if typing.TYPE_CHECKING:
        __call__ = forward


class Branch(nn.Module, Module[P, dict[str, T]]):
    """Module that executes each branch and returns a dictionary with the results of each."""

    def __init__(self, **named_branches: Module[P, T]) -> None:
        super().__init__()
        self.named_branches = named_branches
        self.named_branches = nn.ModuleDict(self.named_branches)  # type: ignore

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, T]:
        outputs: dict[str, T] = {}
        # note: could potentially have each branch on a different cuda device?
        # or maybe we could put each branch in a different cuda stream?
        for name, branch in self.named_branches.items():
            branch_output = branch(*args, **kwargs)
            outputs[name] = branch_output
        return outputs

    if typing.TYPE_CHECKING:
        __call__ = forward


class Merge(nn.Module, Module[[tuple[Tensor, ...] | dict[str, Tensor]], OutT]):
    """Unpacks the output of the previous block (Branch) before it is fed to the wrapped module."""

    __match_args__ = ("f",)

    def __init__(self, f: Module[..., OutT]) -> None:
        """Unpacks the output of a previous block before it is fed to `f`."""
        super().__init__()
        self.f = f

    def forward(self, packed_inputs: tuple[Tensor, ...] | dict[str, Tensor]) -> OutT:
        if isinstance(packed_inputs, (tuple, list)):
            return self.f(*packed_inputs)  # type: ignore
        else:
            return self.f(**packed_inputs)  # type: ignore

    if typing.TYPE_CHECKING:
        __call__ = forward


class Sample(Lambda, Module[[torch.distributions.Distribution], Tensor]):
    def __init__(self, differentiable: bool = False) -> None:
        super().__init__(
            f=operator.methodcaller("rsample" if differentiable else "sample")
        )
        self._differentiable = differentiable

    @property
    def differentiable(self) -> bool:
        return self._differentiable

    @differentiable.setter
    def differentiable(self, value: bool) -> None:
        self._differentiable = value
        self.f = operator.methodcaller("rsample" if value else "sample")

    def forward(self, dist: torch.distributions.Distribution) -> Tensor:
        return super().forward(dist)

    def extra_repr(self) -> str:
        return f"differentiable={self.differentiable}"

    if typing.TYPE_CHECKING:
        __call__ = forward


class SampleIfDistribution(
    nn.Module, Module[[Union[Tensor, torch.distributions.Distribution]], Tensor]
):
    def __init__(self, differentiable=False) -> None:
        super().__init__()
        self._differentiable = differentiable
        self.sample = Sample(differentiable=differentiable)

    @property
    def differentiable(self) -> bool:
        return self._differentiable

    @differentiable.setter
    def differentiable(self, value: bool):
        self.sample.differentiable = value
        self._differentiable = value

    # TODO: add __init__ and such
    def forward(
        self, tensor_or_distribution: Tensor | torch.distributions.Distribution
    ) -> Tensor:
        if isinstance(tensor_or_distribution, torch.distributions.Distribution):
            return self.sample(tensor_or_distribution)
        return tensor_or_distribution


def independent_normal_layer(
    scale: Tensor | float = 1.0,
    differentiable: bool = True,
    reinterpreted_batch_ndims: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        Lambda(torch.distributions.Normal, scale=scale),
        Lambda(torch.distributions.Independent, reinterpreted_batch_ndims=1),
        Sample(differentiable=differentiable),
    )

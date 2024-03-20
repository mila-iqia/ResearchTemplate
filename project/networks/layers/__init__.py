from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any, overload

from torch import (
    Tensor,  # noqa: F401
    nn,
)
from torch._jit_internal import _copy_to_script_wrapper
from typing_extensions import TypeVar

from project.utils.types import Module

ModuleType = TypeVar("ModuleType", bound=Module[..., Any], default=Module[[Tensor], Tensor])


class Sequential(nn.Sequential, Sequence[ModuleType]):
    # Small typing fixes for torch.nn.Sequential

    _modules: dict[str, ModuleType]

    @overload
    def __init__(self, *args: ModuleType) -> None: ...

    @overload
    def __init__(self, **kwargs: ModuleType) -> None: ...

    @overload
    def __init__(self, arg: dict[str, ModuleType]) -> None: ...

    def __init__(self, *args, **kwargs):
        if args:
            assert not kwargs, "can only use *args or **kwargs, not both"
            if len(args) == 1 and isinstance(args[0], dict):
                new_args = (OrderedDict(args[0]),)
            else:
                new_args = []
                for arg in args:
                    if not isinstance(arg, nn.Module) and callable(arg):
                        from project.algorithms.common.layers import Lambda

                        arg = Lambda(arg)
                    new_args.append(arg)
            args = new_args

        if kwargs:
            assert not args, "can only use *args or **kwargs, not both"

            from project.algorithms.common.layers import Lambda

            new_kwargs = {}
            for name, module in kwargs.items():
                if not isinstance(module, nn.Module) and callable(module):
                    from project.algorithms.common.layers import Lambda

                    module = Lambda(module)
                new_kwargs[name] = module
            kwargs = new_kwargs

            args = (OrderedDict(kwargs),)

        super().__init__(*args)
        self._modules

    @overload
    def __getitem__(self, idx: int) -> ModuleType: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequential[ModuleType]: ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int | slice) -> Sequential[ModuleType] | ModuleType:
        if isinstance(idx, slice):
            # NOTE: Fixing this here, subclass constructors shouldn't be called on getitem with
            # slice.
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __iter__(self) -> Iterator[ModuleType]:
        return super().__iter__()  # type: ignore

    def __setitem__(self, idx: int, module: ModuleType) -> None:
        # Violates the LSP, but eh.
        return super().__setitem__(idx, module)

    def forward(self, *args, **kwargs):
        out = None
        for i, module in enumerate(self):
            if i == 0:
                out = module(*args, **kwargs)  # type: ignore
            else:
                out = module(out)  # type: ignore
        assert out is not None
        return out

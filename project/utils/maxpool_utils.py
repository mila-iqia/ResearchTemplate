from __future__ import annotations

import functools
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from project.algorithms.common.hooks import named_modules_of_type
from project.utils.types import Module

_MaxPoolLayer = nn.modules.pooling._MaxPoolNd | nn.modules.pooling._AdaptiveMaxPoolNd


@contextmanager
def save_maxpool_indices(
    module_or_modules: Module | Mapping[str, _MaxPoolLayer],
) -> Generator[dict[str, Tensor], None, None]:
    """Saves maxpool indices from nn.MaxPool layers (if any) so they can be passed to the decoder.

    This can be used in conjunction with `use_indices_in_maxunpool` to avoid having to pass around
    the indices from the `nn.MaxPool2d` layers to the `nn.MaxUnpool2d` layers in the decoder.

    Here's an example:

    ```python
    with save_maxpool_indices(encoder_network) as maxpool_indices:
        latent = encoder_network(input)

    with use_indices_in_maxunpool(decoder_network, maxpool_indices):
        output = decoder_network(latent)
    ```

    Parameters
    ------------
    - module_or_modules: A module whose MaxPool layers should have their indices saved, or a \
        mapping from module names to maxpool layers.
    Yields
    ------
    A dictionary mapping from module name to the maxpool indices for that module.
    """
    maxpool_indices: dict[str, Tensor] = {}
    returned_indices: dict[str, bool] = {}
    hook_handles: list[RemovableHandle] = []

    if isinstance(module_or_modules, nn.Module | Module):
        named_maxpool_modules = dict(
            named_modules_of_type(
                module_or_modules,
                (nn.modules.pooling._MaxPoolNd, nn.modules.pooling._AdaptiveMaxPoolNd),
            )
        )
    else:
        named_maxpool_modules = module_or_modules

    for module_name, module in named_maxpool_modules.items():
        # NOTE: Need to set it back to its original value after the hook.
        returned_indices[module_name] = module.return_indices
        # Set this so the module's `forward` (temporarily) also returns the maxpool indices.
        module.return_indices = True

        hook_handle = module.register_forward_hook(
            functools.partial(
                _save_maxpool_indices_forward_hook,
                module_name,
                maxpool_indices,
                returned_indices[module_name],
            ),
            with_kwargs=True,
        )
        hook_handles.append(hook_handle)

    yield maxpool_indices

    # Reset the "return_indices" attribute to its original value.
    for name, module in named_maxpool_modules.items():
        module.return_indices = returned_indices.pop(name)

    # Remove the hooks
    for handle in hook_handles:
        handle.remove()


@contextmanager
def use_indices_in_maxunpool(
    module_or_modules: Module | Mapping[str, nn.modules.pooling._MaxUnpoolNd],
    maxpool_indices: dict[str, Tensor] | list[Tensor],
    consume_indices: bool = False,
) -> Generator[None, None, None]:
    """Passes any saved MaxPool indices from the encoder to the MaxUnpool layers of the decoder.

    This can be used in conjunction with `save_maxpool_indices` to avoid having to pass around
    some arguments between encoding-decoding (forward/backward) passes.

    Here's an example:

    ```python
    with save_maxpool_indices(encoder_network) as maxpool_indices:
        latent = encoder_network(input)

    with use_indices_in_maxunpool(decoder_network, maxpool_indices):
        output = decoder_network(latent)
    ```

    Arguments:
    ------------
    - module_or_modules: The module whose MaxUnpool layers should use the maxpool indices.
    - maxpool_indices: The maxpool indices saved from the forward pass with the encoder network.
    - consume_indices: If `True`, then the maxpool indices are consumed from the dict.
    """
    # add forward_pre hooks that add the maxpool indices to the input of the nn.MaxUnpool2d layers
    hook_handles: list[RemovableHandle] = []

    if isinstance(module_or_modules, nn.Module | Module):
        named_maxunpool_modules = dict(
            named_modules_of_type(module_or_modules, nn.modules.pooling._MaxUnpoolNd)
        )
    else:
        named_maxunpool_modules = dict(module_or_modules)

    if isinstance(maxpool_indices, list):
        # We need as many maxpool indices as there are MaxUnpool layers.
        assert len(maxpool_indices) == len(named_maxunpool_modules)
        maxpool_indices = dict(zip(named_maxunpool_modules.keys(), maxpool_indices))

    for module_name, module in named_maxunpool_modules.items():
        if has_maxunpool2d(module):
            assert module_name in maxpool_indices, (module_name, maxpool_indices.keys())
        elif module_name not in maxpool_indices:
            # NOTE: If the module isn't in the maxpool_indices dict, then it must be because the
            # encoder network with the nn.MaxPool layer was already returning the maxpool indices.
            continue
        # NOTE: Assuming that the same MaxUnpool2d layer is only used once in a given forward pass.
        indices = (
            maxpool_indices.pop(module_name) if consume_indices else maxpool_indices[module_name]
        )
        hook_handle = module.register_forward_pre_hook(
            functools.partial(  # type: ignore (the type of args are specific to MaxUnpool2d)
                _use_maxpool_indices_in_maxunpool_forward_pre_hook,
                indices,
            ),
            with_kwargs=True,
        )
        hook_handles.append(hook_handle)

    yield

    # Remove the hooks
    for handle in hook_handles:
        handle.remove()

    # remove hooks


def _save_maxpool_indices_forward_hook(
    _name: str,
    _maxpool_indices: dict[str, Tensor],
    _did_return_indices: bool,
    /,
    module: nn.MaxPool2d | nn.AdaptiveMaxPool2d,
    args: tuple[Tensor, ...],
    kwargs: Any,
    output: tuple[Tensor, Tensor],
):
    out, indices = output

    # There shouldn't already have an entry at this key, unless perhaps the net is used twice in
    # the same 'with' block? I don't think we want to encourage doing that anyway (just use two
    # with blocks instead to get two different dictionaries, instead of a single changing dict).
    assert _name not in _maxpool_indices
    _maxpool_indices[_name] = indices

    if _did_return_indices:
        # If the network originally returned the maxpool indices, then we return them here too so
        # we don't break things.
        return out, indices
    else:
        return out


def _use_maxpool_indices_in_maxunpool_forward_pre_hook(
    _maxpool_indices: Tensor,
    module: nn.modules.pooling._MaxUnpoolNd,
    args: tuple[Tensor] | tuple[Tensor, list[int]],
    kwargs: dict[str, Any],
) -> tuple[tuple[Tensor], dict[str, Any]]:
    """Sets indices=`_maxpool_indices` in the forward method of the given MaxUnpool module."""
    # NOTE: The MaxUnpool layer shouldn't already be receiving the indices as an argument, because
    # we only add this hook to the maxunpool layers whose corresponding forward (maxpool) layer
    # didn't already return the maxpool indices, and therefore rely on this hook to provide the
    # maxpool indices to the maxunpool layers.
    input: Tensor
    new_args: tuple[Tensor]
    if not kwargs:
        # NOTE: Perhaps some subclasses of nn.MaxUnpool2d would have additional positional input
        # args? IF so, then this would make it work:
        # assert len(args) >= 1
        # out_args = (input, *args[1:])
        assert len(args) == 1
        input = args[0]
        new_args = (input,)

    else:
        assert "indices" not in kwargs  # see note above.
        input = kwargs.pop("input")
        assert isinstance(input, Tensor)
        new_args = (input,)

    new_kwargs = kwargs | {"indices": _maxpool_indices}
    return new_args, new_kwargs


def has_maxpool2d(network: nn.Module) -> bool:
    for module in network.modules():
        if isinstance(module, nn.MaxPool2d):
            return True
    return False


def has_maxunpool2d(network: nn.Module | Module) -> bool:
    for module in network.modules():
        if isinstance(module, nn.MaxUnpool2d):
            return True
    return False

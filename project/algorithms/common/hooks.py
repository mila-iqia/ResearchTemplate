from __future__ import annotations

import contextlib
import functools
import typing
from collections.abc import Generator, Iterable
from functools import partial
from typing import Any, Concatenate, overload

import torch
from torch import Tensor, nn
from torch import distributions as dist
from torch.utils.hooks import RemovableHandle

from project.algorithms.common.layers import (
    OutT,
    Sample,
    SampleIfDistribution,
    T,
)
from project.utils.types import Module, is_sequence_of

if typing.TYPE_CHECKING:
    from project.networks.layers import Sequential


@contextlib.contextmanager
def get_distributions(
    network: nn.Module,
) -> Generator[list[dist.Distribution], None, None]:
    yielded = []
    with get_module_inputs(named_modules_of_type(network, Sample)) as distributions:
        yield yielded
    yielded.extend(list(distributions.values()))


def modules_of_type[ModuleType: Module](
    module: nn.Module, module_type: type[ModuleType] | tuple[type[ModuleType], ...]
) -> Iterable[ModuleType]:
    for mod in module.modules():
        if isinstance(mod, module_type):
            yield mod


def named_modules_of_type[ModuleType: Module](
    module: Module, module_type: type[ModuleType] | tuple[type[ModuleType], ...]
) -> Iterable[tuple[str, ModuleType]]:
    for name, mod in module.named_modules():
        if isinstance(mod, module_type):
            yield name, mod


@contextlib.contextmanager
def get_block_inputs(
    sequential_network: Sequential[Module[[T], Any]] | nn.Sequential | Iterable[Module[[T], Any]],
) -> Generator[list[T], None, None]:
    yielded: list[T] = []
    named_layers = [(f"{i}", layer) for i, layer in enumerate(sequential_network)]
    with get_module_inputs(named_layers) as inputs:
        # NOTE: This is in the same order as the layers since dicts are ordered.
        # yield list(inputs.values())
        yield yielded
    yielded.extend([inputs[f"{i}"] for i in range(len(named_layers))])


@overload
@contextlib.contextmanager
def get_block_outputs(
    sequential_network: nn.Sequential,
) -> Generator[list[Tensor], None, None]:
    ...


@overload
@contextlib.contextmanager
def get_block_outputs(
    sequential_network: Sequential[Module[..., OutT]],
) -> Generator[list[OutT], None, None]:
    ...


@overload
@contextlib.contextmanager
def get_block_outputs(
    sequential_network: Iterable[Module[..., OutT]],
) -> Generator[list[OutT], None, None]:
    ...


@contextlib.contextmanager
def get_block_outputs(
    sequential_network: Sequential[Module[..., OutT]]
    | nn.Sequential
    | Iterable[Module[..., OutT]],
) -> Generator[list[OutT], None, None]:
    yielded: list[OutT] = []
    named_layers = [(f"{i}", layer) for i, layer in enumerate(sequential_network)]
    with get_module_outputs(named_layers) as outputs:
        # NOTE: This is in the same order as the layers since dicts are ordered.
        # yield list(inputs.values())
        yield yielded
    yielded.extend([outputs[f"{i}"] for i in range(len(named_layers))])


@contextlib.contextmanager
def get_module_inputs(
    layers: Module[[T], OutT] | Iterable[tuple[str, Module[[T], OutT]]],
) -> Generator[dict[str, T], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, nn.Module | Module) else layers  # type: ignore
    layer_inputs: dict[str, T] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_input_hook,
            name,
            layer_inputs,
        )
        hook_handle = layer.register_forward_pre_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_inputs

    for hook_handle in hook_handles:
        hook_handle.remove()


@contextlib.contextmanager
def get_module_outputs(
    layers: Module[..., OutT] | Iterable[tuple[str, Module[..., OutT]]],
) -> Generator[dict[str, OutT], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, Module) else layers  # type: ignore
    layer_outputs: dict[str, OutT] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_output_hook,
            name,
            layer_outputs,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_outputs

    for hook_handle in hook_handles:
        hook_handle.remove()


@contextlib.contextmanager
def save_all_module_inputs_and_outputs(
    layers: Module[Concatenate[T, ...], OutT]
    | Iterable[tuple[str, Module[Concatenate[T, ...], OutT]]],
) -> Generator[tuple[dict[str, T], dict[str, OutT]], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, Module) else layers  # type: ignore
    layer_inputs: dict[str, T] = {}
    layer_outputs: dict[str, OutT] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_input_and_output_hook,
            name,
            layer_inputs,
            layer_outputs,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_inputs, layer_outputs

    for hook_handle in hook_handles:
        hook_handle.remove()


def _save_input_and_output_hook(
    _layer_name: str,
    _inputs: dict[str, T],
    _outputs: dict[str, OutT],
    /,
    module: Module[Concatenate[T, ...], OutT],
    args: tuple[T, ...],
    kwargs: dict[str, Any],
    output: OutT,
):
    """Saves the layer inputs and outputs in the given dictionaries.

    This should only really be used with a `functools.partial` that pre-binds the layer name,
    as well as the input and output dictionaries.

    NOTE: This only saves the first input to the network.
    """
    if len(args) >= 1:
        _inputs[_layer_name] = args[0]
    _outputs[_layer_name] = output


def _save_input_hook(
    _layer_name: str,
    _inputs: dict[str, T],
    /,
    module: Module[Concatenate[T, ...], OutT],
    args: tuple[T, ...],
    kwargs: dict[str, Any],
):
    """Saves the layer inputs and outputs in the given dictionaries.

    This should only really be used with a `functools.partial` that pre-binds the layer name,
    as well as the input and output dictionaries.

    NOTE: This only saves the first input to the network.
    """
    assert len(args) == 1, "TODO: Assumes a single input per layer atm."
    _inputs[_layer_name] = args[0]


def _save_output_hook(
    _layer_name: str,
    _outputs: dict[str, OutT],
    /,
    module: Module[Concatenate[T, ...], OutT],
    args: tuple[T, ...],
    kwargs: dict[str, Any],
    output: OutT,
):
    """Saves the layer inputs and outputs in the given dictionaries.

    This should only really be used with a `functools.partial` that pre-binds the layer name,
    as well as the input and output dictionaries.

    NOTE: This only saves the first input to the network.
    """
    _outputs[_layer_name] = output


def _replace_layer_input_hook(
    _input_to_use: Tensor,
    module: nn.Module,
    args: tuple[Tensor, ...],
):
    assert len(args) == 1
    return (_input_to_use,)
    replaced_inputs = _input_to_use
    if not isinstance(replaced_inputs, tuple):
        replaced_inputs = (replaced_inputs,)
    assert len(replaced_inputs) == len(args)
    assert all(
        replaced_input.shape == arg.shape for replaced_input, arg in zip(replaced_inputs, args)
    )
    return replaced_inputs


@contextlib.contextmanager
def set_layer_inputs(
    layers_and_inputs: dict[nn.Module, Tensor],
):
    """Context that temporarily makes these layers use these inputs in their forward pass."""
    handles: list[RemovableHandle] = []
    for layer, input in layers_and_inputs.items():
        handle = layer.register_forward_pre_hook(partial(_replace_layer_input_hook, input))
        handles.append(handle)

    yield

    for handle in handles:
        handle.remove()


@contextlib.contextmanager
def _save_inputs_and_outputs(
    layers: Module[[T], OutT] | Iterable[tuple[str, Module[[T], OutT]]],
) -> Generator[tuple[dict[str, T], dict[str, OutT]], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, Module) else layers  # type: ignore
    layer_inputs: dict[str, T] = {}
    layer_outputs: dict[str, OutT] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_input_and_output_hook,
            name,
            layer_inputs,
            layer_outputs,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_inputs, layer_outputs

    for hook_handle in hook_handles:
        hook_handle.remove()


@contextlib.contextmanager
def save_forward_activation_distributions(
    module: nn.Module,
) -> Generator[dict[str, torch.distributions.Distribution], None, None]:
    from .layers import Sample

    named_sampling_layers = dict(named_modules_of_type(module, Sample))
    with save_all_module_inputs_and_outputs(named_sampling_layers.items()) as (
        layer_inputs,
        _,
    ):
        yield layer_inputs


def _get_input_and_output_shapes_hook(
    _name: str,
    _input_shapes: dict[str, tuple[int, ...]],
    _output_shapes: dict[str, tuple[int, ...]],
    _shape_starting_from_dim: int,
    /,
    module: Module,
    inputs: tuple[Any, ...] | Any,
    input_kwargs: dict[str, Any],
    outputs: Tensor | Any,
) -> None:
    """Hook that saves the shape of the (first) input tensor and of the output tensor."""
    assert isinstance(inputs, tuple)
    if inputs and isinstance(inputs[0], Tensor) and not input_kwargs:
        _input_shapes[_name] = inputs[0].shape[_shape_starting_from_dim:]
    if isinstance(outputs, Tensor):
        _output_shapes[_name] = outputs.shape[_shape_starting_from_dim:]


@contextlib.contextmanager
def get_input_and_output_shapes(
    module: Module,
) -> Generator[tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]], None, None]:
    input_shapes: dict[str, tuple[int, ...]] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}

    hook_handles: list[RemovableHandle] = []

    for name, layer in module.named_modules():
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _get_input_and_output_shapes_hook,
            name,
            input_shapes,
            output_shapes,
            1,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield input_shapes, output_shapes

    for hook_handle in hook_handles:
        hook_handle.remove()


@contextlib.contextmanager
def save_input_output_shapes(
    module: nn.Module,
) -> Generator[tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]], None, None]:
    """Makes the module easier to "invert" by adding a hook to each layer that sets its
    `input_shape` and `output_shape` attributes during a forward pass.

    Modifies the module in-place.
    """
    input_shapes: dict[str, tuple[int, ...]] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}

    hook_handles: list[RemovableHandle] = []

    for name, layer in module.named_modules():
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _get_input_and_output_shapes_hook,
            name,
            input_shapes,
            output_shapes,
            1,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield input_shapes, output_shapes

    for hook_handle in hook_handles:
        hook_handle.remove()

    for name, input_shape in input_shapes.items():
        submodule = module
        for attr in name.split(".") if name else []:
            submodule = getattr(submodule, attr)
        submodule.input_shape = tuple(input_shape)  # type: ignore

    for name, output_shape in output_shapes.items():
        submodule = module
        for attr in name.split(".") if name else []:
            submodule = getattr(submodule, attr)
        submodule.output_shape = tuple(output_shape)  # type: ignore


@functools.singledispatch
def detach(value: Any) -> Any:
    try:
        return value.detach()
    except AttributeError as err:
        raise NotImplementedError(
            f"Don't know how to detach values of type {type(value)}."
        ) from err


@detach.register(Tensor)
def _detach_tensor(value: Tensor) -> Tensor:
    return value.detach()


@detach.register(torch.distributions.Normal)
def _detach_distribution(value: torch.distributions.Normal) -> Tensor:
    return type(value)(loc=detach(value.loc), scale=detach(value.scale))  # type: ignore


def _detach_block_inputs_forward_pre_hook(
    module: nn.Module,
    inputs: tuple[Tensor | torch.distributions.Distribution, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Tensor | torch.distributions.Distribution, ...], dict[str, Any]]:
    assert isinstance(inputs, tuple)
    if isinstance(module, Sample | SampleIfDistribution):
        if len(inputs) != 1 or kwargs:
            raise NotImplementedError(f"Don't know how to detach {inputs}.")
        input: Tensor | torch.distributions.Distribution = inputs[0]
        if isinstance(input, Tensor):
            return (input.detach(),), kwargs
        if not module.differentiable:
            return inputs, kwargs
        # Need to detach a distribution!
        return (detach(input),), kwargs

    assert is_sequence_of(inputs, Tensor)

    detached_inputs = tuple(t.detach() for t in inputs)
    detached_kwargs = {
        key: value.detach() if isinstance(value, Tensor) else value
        for key, value in kwargs.items()
    }
    return detached_inputs, detached_kwargs


@contextlib.contextmanager
def detach_input_to_each_blocks(
    sequential_network: Sequential[Module[[T], Any]] | nn.Sequential,
):
    hook_handles: list[RemovableHandle] = []
    for block in sequential_network.children():
        hook = block.register_forward_pre_hook(
            _detach_block_inputs_forward_pre_hook, with_kwargs=True
        )
        hook_handles.append(hook)

    yield

    for hook in hook_handles:
        hook.remove()

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Concatenate, Literal, ParamSpec

import flax.core
import flax.linen
import flax.struct
import jax
import jax.experimental
import torch  # noqa
from jax._src.sharding_impls import UNSPECIFIED, Device, UnspecifiedValue
from typing_extensions import TypeVar

P = ParamSpec("P")
Out = TypeVar("Out", covariant=True)


@functools.wraps(jax.jit)
def jit(
    fn: Callable[P, Out],
    in_shardings: UnspecifiedValue = UNSPECIFIED,
    out_shardings: UnspecifiedValue = UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> Callable[P, Out]:
    # Small type hint fix for jax's `jit` (preserves the signature of the callable).
    # TODO: Remove once [our PR to Jax](https://github.com/jax-ml/jax/pull/23720) is merged

    return jax.jit(
        fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )


In = TypeVar("In")
Aux = TypeVar("Aux")


@functools.wraps(jax.value_and_grad)
def value_and_grad(
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Literal[0] = 0,
    has_aux: Literal[True] = True,
) -> Callable[Concatenate[In, P], tuple[tuple[Out, Aux], In]]:
    # Small type hint fix for jax's `value_and_grad` (preserves the signature of the callable).
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)  # type: ignore


@functools.wraps(flax.struct.field)
def field(
    pytree_node=True,
    _field_fn: Callable[P, dataclasses.Field] = dataclasses.field,
    *args: P.args,
    **kwargs: P.kwargs,
):
    # Typing fix for `flax.struct.field` so that it doesn't drop the signature of the
    # `dataclasses.field` function that it calls.
    metadata = kwargs.get("metadata")
    assert metadata is None or isinstance(metadata, dict)
    if metadata is None:
        metadata = {}
    metadata.setdefault("pytree_node", pytree_node)
    return _field_fn(*args, **kwargs)

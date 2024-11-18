"""Small typing helpers for Jax.

This makes `jax.jit` preserve the signature of the wrapped callable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Concatenate, Literal, ParamSpec, overload

import jax
import jax.experimental
from jax._src.sharding_impls import UNSPECIFIED, Device, UnspecifiedValue
from typing_extensions import TypeVar

P = ParamSpec("P")
Out = TypeVar("Out", covariant=True)


# @functools.wraps(jax.jit)
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


# @functools.wraps(jax.value_and_grad)
def value_and_grad(
    fn: Callable[Concatenate[In, P], tuple[Out, Aux]],
    argnums: Literal[0] = 0,
    has_aux: Literal[True] = True,
) -> Callable[Concatenate[In, P], tuple[tuple[Out, Aux], In]]:
    # Small type hint fix for jax's `value_and_grad` (preserves the signature of the callable).
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)  # type: ignore


_T = TypeVar("_T")


# @functools.wraps(flax.struct.field)
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(
    *,
    default: _T,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    pytree_node: bool = True,
) -> _T: ...
@overload
def field(
    *,
    default_factory: Callable[[], _T],
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    pytree_node: bool = True,
) -> _T: ...
@overload
def field(
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    pytree_node: bool = True,
) -> Any: ...


def field(
    *,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only=dataclasses.MISSING,
    pytree_node: bool | None = None,
):
    """Small Typing fix for `flax.struct.field`.

    - Add type annotations so it doesn't drop the signature of the `dataclasses.field` function.
    - Make the `pytree_node` has a default value of `False` for ints and bools, and `True` for
      everything else.
    """
    if pytree_node is None and isinstance(default, int):  # note: also includes `bool`.
        pytree_node = False
    if pytree_node is None:
        pytree_node = True
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("pytree_node", pytree_node)
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )  # type: ignore

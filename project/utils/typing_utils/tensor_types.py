"""Inspired from https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/tensor_typing.py"""

from __future__ import annotations

import os
from logging import getLogger
from typing import TypeVar

from beartype import beartype
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped, PyTree
from torch import Tensor

logger = getLogger(__name__)
T = TypeVar("T")


def identity(t: T) -> T:
    return t


# jaxtyping is a misnomer, works for pytorch


class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)


# use env variable TYPECHECK to control whether to use beartype + jaxtyping
should_typecheck = os.environ.get("TYPECHECK", "").lower() in ("false", "0")

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else identity

if should_typecheck:
    logger.info("Type checking is enabled.")
else:
    logger.info("Type checking is disabled.")

__all__ = [
    Shaped,
    Float,
    Int,
    Bool,
    typecheck,
    should_typecheck,
    beartype_isinstance,
]

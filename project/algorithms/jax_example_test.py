from typing import ClassVar

import flax
import flax.linen
import torch

from project.algorithms.jax_example import JaxExample

from .testsuites.algorithm_tests import AlgorithmTests


class TestJaxExample(AlgorithmTests[JaxExample]):
    """This algorithm only works with Jax modules."""

    unsupported_network_types: ClassVar[list[type]] = [torch.nn.Module]
    _supported_network_types: ClassVar[list[type]] = [flax.linen.Module]

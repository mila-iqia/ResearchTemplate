from typing import ClassVar

import flax
import flax.linen
import torch

from project.algorithms.jax_example import JaxAlgorithm

from .algorithm_test import AlgorithmTests


class TestJaxAlgorithm(AlgorithmTests[JaxAlgorithm]):
    """This algorithm only works with Jax modules."""

    algorithm_name: ClassVar[str] = "jax_algo"
    unsupported_network_types: ClassVar[list[type]] = [torch.nn.Module]
    _supported_network_types: ClassVar[list[type]] = [flax.linen.Module]

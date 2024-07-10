from typing import ClassVar

import torch

from project.algorithms.testsuites.classification_tests import ClassificationAlgorithmTests

from .example import ExampleAlgorithm


class TestExampleAlgorithm(ClassificationAlgorithmTests[ExampleAlgorithm]):
    algorithm_type = ExampleAlgorithm
    algorithm_name: str = "example"
    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]
    _supported_network_types: ClassVar[list[type]] = [torch.nn.Module]

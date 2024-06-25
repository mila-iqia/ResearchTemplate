from typing import ClassVar

import torch

from project.algorithms.bases.image_classification_test import ClassificationAlgorithmTests

from .manual_optimization_example import ManualGradientsExample


class TestManualOptimizationExample(ClassificationAlgorithmTests[ManualGradientsExample]):
    algorithm_type = ManualGradientsExample
    algorithm_name: str = "manual_optimization"

    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]
    _supported_network_types: ClassVar[list[type]] = [torch.nn.Module]

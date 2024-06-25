from typing import ClassVar

import torch

from project.algorithms.classification_tests import ClassificationAlgorithmTests
from project.datamodules.vision import VisionDataModule

from .manual_optimization_example import ManualGradientsExample


class TestManualOptimizationExample(ClassificationAlgorithmTests[ManualGradientsExample]):
    algorithm_type = ManualGradientsExample
    algorithm_name: str = "manual_optimization"

    _supported_datamodule_types: ClassVar[list[type]] = [VisionDataModule]
    _supported_network_types: ClassVar[list[type]] = [torch.nn.Module]

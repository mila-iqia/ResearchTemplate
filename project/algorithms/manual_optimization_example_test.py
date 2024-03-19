from typing import ClassVar

from project.algorithms.bases.image_classification_test import ImageClassificationAlgorithmTests

from .manual_optimization_example import ManualGradientsExample


class TestManualOptimizationExample(ImageClassificationAlgorithmTests[ManualGradientsExample]):
    algorithm_type = ManualGradientsExample
    algorithm_name: str = "manual_optimization"

    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]

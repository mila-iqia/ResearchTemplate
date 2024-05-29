from typing import ClassVar

from project.algorithms.bases.image_classification_test import ImageClassificationAlgorithmTests

from .example_algo import ExampleAlgorithm


class TestExampleAlgorithm(ImageClassificationAlgorithmTests[ExampleAlgorithm]):
    algorithm_type = ExampleAlgorithm
    algorithm_name: str = "example_algo"
    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]

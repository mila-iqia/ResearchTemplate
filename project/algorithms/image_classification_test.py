from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeVar
import pytest
from torch import Tensor

from project.algorithms.algorithm_test import AlgorithmTests
from project.algorithms.image_classification import ImageClassificationAlgorithm
from project.datamodules.image_classification import ImageClassificationDataModule
from project.experiment import setup_experiment
from main import run

from .algorithm_test import slow, get_experiment_config

ImageAlgorithmType = TypeVar("ImageAlgorithmType", bound=ImageClassificationAlgorithm)


class ImageClassificationAlgorithmTests(AlgorithmTests[ImageAlgorithmType]):
    metric_name: ClassVar[str] = "train/cross_entropy"
    """The main 'loss' metric to inspect to check if training is working."""

    def test_output_shapes(
        self,
        algorithm: ImageAlgorithmType,
        training_batch: tuple[Tensor, Tensor],
    ):
        """Tests that the output of the algorithm has the correct shape."""
        x, y = training_batch
        output = algorithm(x)
        if isinstance(output, tuple):
            y_pred, x_hat = algorithm(x)
        else:
            y_pred = output
        assert y_pred.shape == (y.shape[0], algorithm.datamodule.num_classes)

    @pytest.fixture(scope="class")
    def training_batch(self, datamodule: ImageClassificationDataModule) -> tuple[Tensor, Tensor]:
        """Returns a batch of data from the training set of the datamodule."""
        datamodule.prepare_data()
        datamodule.setup("fit")
        return next(iter(datamodule.train_dataloader()))

    @slow()
    def test_overfit_single_batch(
        self, network_name: str, datamodule_name: str, tmp_path: Path
    ) -> None:
        """Test the performance of this algorithm when learning from a single batch.

        If this doesn't work, there isn't really a point in trying to train for longer.
        """
        # Number of training iterations (NOTE: each iteration is one call to training_step, which
        # itself may do more than a single update, e.g. in the case of DTP).
        # By how much the model should be better than chance accuracy to pass this test.
        num_training_iterations = 10

        # FIXME: This threshold is really low, we should expect more like > 90% accuracy, but it's
        # currently taking a long time to get those values.
        better_than_chance_threshold_pct = 0.10

        algorithm_name = self.algorithm_name or self.algorithm_cls.__name__.lower()
        assert isinstance(algorithm_name, str)
        assert isinstance(datamodule_name, str)
        assert isinstance(network_name, str)
        experiment_config = get_experiment_config(
            command_line_overrides=[
                f"algorithm={algorithm_name}",
                f"network={network_name}",
                f"datamodule={datamodule_name}",
                "seed=123",
                # "trainer.detect_anomaly=true",
                "~trainer/logger",
                "trainer/callbacks=no_checkpoints",
                "+trainer.overfit_batches=1",
                "+trainer.limit_val_batches=0",
                "+trainer.limit_test_batches=0",
                "+trainer.enable_checkpointing=false",
                f"++trainer.max_epochs={num_training_iterations}",
                f"++trainer.default_root_dir={tmp_path}",
            ]
        )
        assert experiment_config.trainer["max_epochs"] == num_training_iterations

        experiment = setup_experiment(experiment_config)
        classification_error, metrics = run(experiment)

        assert hasattr(experiment.datamodule, "num_classes")
        num_classes: int = experiment.datamodule.num_classes  # type: ignore
        chance_accuracy = 1 / num_classes
        assert classification_error is not None
        accuracy = 1 - classification_error
        assert metrics["train/accuracy"] == accuracy

        # NOTE: In this particular case, this error below is the training error, not the validation
        # error.
        assert accuracy > (chance_accuracy + better_than_chance_threshold_pct)

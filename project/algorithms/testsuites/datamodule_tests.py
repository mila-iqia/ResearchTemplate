import abc
import sys
from pathlib import Path
from typing import Generic, TypeVar

import hydra_zen
import matplotlib.pyplot as plt
import omegaconf
import pytest
from lightning import LightningDataModule
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture, get_test_source_and_temp_file_paths
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.tv_tensors import Image

from project.algorithms.testsuites.lightning_module_tests import convert_list_and_tuples_to_dicts
from project.conftest import command_line_overrides, setup_with_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.image_classification.inaturalist import INaturalistDataModule
from project.datamodules.image_classification.mnist import MNISTDataModule

DataModuleType = TypeVar("DataModuleType", bound=LightningDataModule)


class DataModuleTests(Generic[DataModuleType], abc.ABC):
    @pytest.fixture(
        scope="class",
        params=[
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
            RunningStage.TESTING,
            pytest.param(
                RunningStage.PREDICTING,
                marks=pytest.mark.xfail(
                    reason="Might not be implemented by the datamodule.",
                    raises=MisconfigurationException,
                ),
            ),
        ],
    )
    def stage(self, request: pytest.FixtureRequest):
        return getattr(request, "param", RunningStage.TRAINING)

    @pytest.fixture(scope="class")
    def datamodule(self, dict_config: omegaconf.DictConfig) -> DataModuleType:
        """Fixture that creates the datamodule instance, given the current Hydra config."""
        datamodule = hydra_zen.instantiate(dict_config["datamodule"])
        return datamodule

    @pytest.fixture(scope="class")
    def dataloader(self, datamodule: DataModuleType, stage: RunningStage) -> DataLoader:
        datamodule.prepare_data()
        if stage == RunningStage.TRAINING:
            datamodule.setup("fit")
            dataloader = datamodule.train_dataloader()
        elif stage in [RunningStage.VALIDATING, RunningStage.SANITY_CHECKING]:
            datamodule.setup("validate")
            dataloader = datamodule.val_dataloader()
        elif stage == RunningStage.TESTING:
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        else:
            assert stage == RunningStage.PREDICTING
            datamodule.setup("predict")
            dataloader = datamodule.predict_dataloader()
        return dataloader

    @pytest.fixture(scope="class")
    def batch(self, dataloader: DataLoader):
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch

    def test_first_batch(
        self,
        batch,
        tensor_regression: TensorRegressionFixture,
    ):
        batch = convert_list_and_tuples_to_dicts(batch)
        tensor_regression.check(batch, include_gpu_name_in_stats=False)


# @pytest.mark.slow

# Use a dummy, empty algorithm, to keep the datamodule tests independent of the algorithms.
# This is a unit test for the datamodule, so we don't want to involve the algorithm here.

ImageClassificationDataModuleType = TypeVar(
    "ImageClassificationDataModuleType", bound=ImageClassificationDataModule
)


@pytest.mark.parametrize(command_line_overrides.__name__, ["algorithm=no_op"], indirect=True)
class ImageClassificationDataModuleTests(DataModuleTests[ImageClassificationDataModuleType]):
    def test_first_batch(  # type: ignore
        self,
        datamodule: ImageClassificationDataModuleType,
        batch: tuple[Image, Tensor],
        request: pytest.FixtureRequest,
        tensor_regression: TensorRegressionFixture,
        original_datadir: Path,
        stage: RunningStage,
        datadir: Path,
    ):
        super().test_first_batch(batch, tensor_regression)

        n_rows = 4
        n_cols = 4
        if (
            len(batch) != 2
            or (x := batch[0]).ndim != 4
            or x.shape[1] not in (1, 3)
            or x.shape[0] < n_rows * n_cols
        ):
            return

        x, y = batch

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
        for i in range(n_rows):
            for j in range(n_cols):
                index = i * n_cols + j
                axis: plt.Axes = axes[i, j]  # type: ignore

                image = x[index].permute(1, 2, 0)  # to channels last.
                if image.shape[-1] == 5:
                    # moving mnist. Keep only 3 frames to make a figure with.
                    image = image[:, :, :3]
                axis.imshow(image)
                axis.axis("off")
                if y.ndim == 1:
                    label = y[index].item()
                    axis.set_title(f"{index=} {label=}")
                else:
                    # moving mnist, y isn't a label, it's another image.
                    axis.set_title(f"{index=}")

        split = {
            RunningStage.TRAINING: "training",
            RunningStage.VALIDATING: "validation",
            RunningStage.TESTING: "test",
            RunningStage.PREDICTING: "prediction(?)",
        }

        fig.suptitle(f"First {split[stage]} batch of datamodule {type(datamodule).__name__}")
        figure_path, _ = get_test_source_and_temp_file_paths(
            extension=".png",
            request=request,
            original_datadir=original_datadir,
            datadir=datadir,
        )
        figure_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(figure_path)
        if "-vvv" in sys.argv:
            plt.waitforbuttonpress(timeout=10)

        gitignore_file = original_datadir / ".gitignore"
        lines_to_add = [
            "# Ignore images generated by tests.",
            "*.png",
            "",
        ]
        original_datadir.mkdir(exist_ok=True, parents=True)
        if not gitignore_file.exists():
            gitignore_file.write_text("\n".join(lines_to_add))
            return

        lines = gitignore_file.read_text().splitlines()
        if not any(line.strip() == "*.png" for line in lines):
            with gitignore_file.open("a") as f:
                f.write(
                    "\n".join(
                        lines_to_add,
                    )
                )


@setup_with_overrides("algorithm=no_op datamodule=mnist")
class TestMNISTDataModule(ImageClassificationDataModuleTests[MNISTDataModule]): ...


@setup_with_overrides("algorithm=no_op datamodule=fashion_mnist")
class TestFashionMNISTDataModule(ImageClassificationDataModuleTests[FashionMNISTDataModule]): ...


@setup_with_overrides("algorithm=no_op datamodule=cifar10")
class TestCIFAR10DataModule(ImageClassificationDataModuleTests[CIFAR10DataModule]): ...


# todo: add the marks from the `conftest` "default_marks_for_config_name" or similar.
@pytest.mark.slow
@setup_with_overrides("algorithm=no_op datamodule=inaturalist")
class TestINaturalistDataModule(ImageClassificationDataModuleTests[INaturalistDataModule]): ...

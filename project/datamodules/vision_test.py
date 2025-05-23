import sys
from logging import getLogger
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import pytest
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture, get_test_source_and_temp_file_paths
from torch import Tensor
from torchvision.tv_tensors import Image

from project.datamodules.datamodule_tests import DataModuleTests
from project.datamodules.vision import VisionDataModule

logger = getLogger(__name__)
VisionDataModuleType = TypeVar("VisionDataModuleType", bound=VisionDataModule)


class VisionDataModuleTests(DataModuleTests[VisionDataModuleType]):
    """Tests for a datamodule/dataset for vision tasks.

    This is a simple data regression test for now.
    For each of the `train_dataloader`, `valid_dataloader`
    """

    def test_first_batch(  # type: ignore
        self,
        datamodule: VisionDataModuleType,
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

        assert (x := batch[0]).ndim == 4
        assert x.shape[1] in (1, 3)

        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch, None
        assert isinstance(x, Tensor)
        assert y is None or isinstance(y, Tensor)

        if x.shape[0] < n_rows * n_cols:
            logger.warning(f"Batch size is too small to generate an {n_rows} by {n_cols} figure!")
            n_rows = 1
            n_cols = x.shape[0]

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
                if y is not None and y.ndim == 1:
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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture, get_test_source_and_temp_file_paths
from torch import Tensor

from project.conftest import command_line_overrides
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.vision import VisionDataModule
from project.utils.testutils import run_for_all_configs_in_group
from project.utils.typing_utils import is_sequence_of


@pytest.mark.slow
@pytest.mark.parametrize(
    "stage",
    [
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
@pytest.mark.parametrize(command_line_overrides.__name__, ["algorithm=no_op"], indirect=True)
@run_for_all_configs_in_group(group_name="datamodule")
def test_first_batch(
    datamodule: LightningDataModule,
    request: pytest.FixtureRequest,
    tensor_regression: TensorRegressionFixture,
    original_datadir: Path,
    stage: RunningStage,
    datadir: Path,
):
    # Note: using dataloader workers in tests can cause issues, since if a test fails, dataloader
    # workers aren't always cleaned up properly.
    if isinstance(datamodule, VisionDataModule) or hasattr(datamodule, "num_workers"):
        datamodule.num_workers = 0  # type: ignore

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

    iterator = iter(dataloader)
    batch = next(iterator)
    from torchvision.tv_tensors import Image

    if isinstance(datamodule, ImageClassificationDataModule):
        assert isinstance(batch, list | tuple) and len(batch) == 2
        # todo: if we tighten this and make it so vision datamodules return Images, then we should
        # have strict asserts here that check that batch[0] is an Image. It doesn't seem to be the case though.
        # assert isinstance(batch[0], Image)
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], torch.Tensor)
    elif isinstance(datamodule, VisionDataModule):
        if isinstance(batch, list | tuple):
            # assert isinstance(batch[0], Image)
            assert isinstance(batch[0], torch.Tensor)
        else:
            assert isinstance(batch, torch.Tensor)
            assert isinstance(batch, Image)

    if isinstance(batch, dict):
        # fixme: leftover from the RL datamodule proof-of-concept.
        if "infos" in batch:
            # todo: fix this, unsupported because of `object` dtype.
            batch.pop("infos")
        tensor_regression.check(batch, include_gpu_name_in_stats=False)
    else:
        assert is_sequence_of(batch, Tensor)
        tensor_regression.check(
            {f"{i}": batch_i for i, batch_i in enumerate(batch)}, include_gpu_name_in_stats=False
        )

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

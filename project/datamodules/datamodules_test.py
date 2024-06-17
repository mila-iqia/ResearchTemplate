import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from tensor_regression.fixture import TensorRegressionFixture, get_test_source_and_temp_file_paths
from torch import Tensor

from project.utils.testutils import run_for_all_datamodules
from project.utils.types import is_sequence_of

from ..utils.types.protocols import DataModule


@pytest.mark.timeout(25, func_only=True)
@run_for_all_datamodules()
def test_first_batch(
    datamodule: DataModule,
    request: pytest.FixtureRequest,
    tensor_regression: TensorRegressionFixture,
    original_datadir: Path,
    datadir: Path,
):
    # todo: skip this test if the dataset isn't already downloaded (for example on the GitHub CI).
    datamodule.prepare_data()
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))
    if isinstance(batch, dict):
        if "infos" in batch:
            # todo: fix this, unsupported because of `object` dtype.
            batch.pop("infos")
        tensor_regression.check(batch)
    else:
        assert is_sequence_of(batch, Tensor)
        tensor_regression.check({f"{i}": batch_i for i, batch_i in enumerate(batch)})

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

    fig.suptitle(f"First batch of datamodule {type(datamodule).__name__}")
    figure_path, _ = get_test_source_and_temp_file_paths(
        extension=".png", request=request, original_datadir=original_datadir, datadir=datadir
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
    lines = gitignore_file.read_text().splitlines()
    if not any(line.strip() == "*.png" for line in lines):
        with gitignore_file.open("a") as f:
            f.write(
                "\n".join(
                    lines_to_add,
                )
            )

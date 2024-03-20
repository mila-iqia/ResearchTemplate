from __future__ import annotations

from pathlib import Path

import pytest
import torch.testing
from pytest_regressions.file_regression import FileRegressionFixture
from torch import Tensor
from torch.utils.data import DataLoader

from project.configs.datamodule import DATA_DIR
from project.utils.tensor_regression import TensorRegressionFixture
from project.utils.types import PhaseStr

from .moving_mnist import MovingMnistDataModule


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


slow_if_not_downloaded = (
    pytest.mark.slow if not (DATA_DIR / "MovingMNIST").exists() else pytest.mark.timeout(5)
)


@slow_if_not_downloaded
class TestMovingMnist:
    """Tests for the MovingMNIST DataModule."""

    def test_data_dir_is_used(self, data_dir: Path, file_regression: FileRegressionFixture):
        datamodule = MovingMnistDataModule(data_dir=data_dir)
        assert datamodule.data_dir == data_dir
        datamodule.prepare_data()
        files_in_dir = list(map(str, data_dir.iterdir()))
        assert files_in_dir

    @pytest.fixture(params=["train", "val", "test"])
    def phase(self, request: pytest.FixtureRequest) -> PhaseStr:
        return request.param

    def _get_dataloader(self, dm: MovingMnistDataModule, phase: PhaseStr) -> DataLoader:
        dm.prepare_data()
        dm.setup()
        dl = (
            dm.train_dataloader()
            if phase == "train"
            else dm.val_dataloader()
            if phase == "val"
            else dm.test_dataloader()
        )
        assert isinstance(dl, DataLoader)
        return dl

    @pytest.fixture()
    def dataloader(self, data_dir: Path, phase: PhaseStr):
        datamodule = MovingMnistDataModule(data_dir=data_dir)
        return self._get_dataloader(datamodule, phase)

    def test_seeding(
        self,
        data_dir: Path,
        tensor_regression: TensorRegressionFixture,
        phase: PhaseStr,
    ):
        dm_1 = MovingMnistDataModule(data_dir=data_dir, seed=42)
        dm_2 = MovingMnistDataModule(data_dir=data_dir, seed=42)
        dm_3 = MovingMnistDataModule(data_dir=data_dir, seed=123)

        x1, y1 = next(iter(self._get_dataloader(dm_1, phase)))
        x2, y2 = next(iter(self._get_dataloader(dm_2, phase)))
        x3, y3 = next(iter(self._get_dataloader(dm_3, phase)))

        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)

        if phase == "test":
            # The test splits is always the same regardless of the seed
            assert torch.isclose(x1, x3).all()
            assert torch.isclose(y1, y3).all()
        else:
            assert not torch.isclose(x1, x3).all()
            assert not torch.isclose(y1, y3).all()

        tensor_regression.check(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x3": x3,
                "y3": y3,
            }
        )

    def test_first_vs_second_epoch_batches(self, data_dir: Path, phase: PhaseStr):
        dm = MovingMnistDataModule(data_dir=data_dir, seed=42)
        dataloader = self._get_dataloader(dm, phase)
        first_epoch_batch: tuple[Tensor, Tensor] | None = None
        second_epoch_batch: tuple[Tensor, Tensor] | None = None
        # TODO: Do we expect the same batch to be yielded for the second epoch?
        for first_epoch_batch in dataloader:
            break
        for second_epoch_batch in dataloader:
            break
        assert first_epoch_batch is not None
        assert second_epoch_batch is not None
        if phase == "train":
            assert not torch.isclose(first_epoch_batch[0], second_epoch_batch[0]).all()
            assert not torch.isclose(first_epoch_batch[1], second_epoch_batch[1]).all()
        else:
            torch.testing.assert_close(first_epoch_batch, second_epoch_batch)

    def test_dims_work(self, data_dir: Path, phase: PhaseStr):
        dm = MovingMnistDataModule(data_dir=data_dir)
        dataloader = self._get_dataloader(dm, phase)

        for before, after in dataloader:
            for x in [before, after]:
                assert isinstance(x, Tensor)
                b, c, h, w = x.shape
                assert b == dataloader.batch_size == dm.batch_size
                assert c == dm.dims[0]
                assert h == dm.dims[1]
                assert w == dm.dims[2]
            break

    # datamodule.

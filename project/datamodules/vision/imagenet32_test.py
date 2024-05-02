import itertools
from pathlib import Path

import pytest

from project.configs.datamodule import SCRATCH

from .imagenet32 import ImageNet32DataModule


@pytest.mark.slow
def test_dataset_download_works(data_dir: Path):
    batch_size = 16
    datamodule = ImageNet32DataModule(
        data_dir=data_dir,
        readonly_datasets_dir=SCRATCH,
        batch_size=batch_size,
        num_images_per_val_class=10,
    )
    assert datamodule.num_images_per_val_class == 10
    assert datamodule.val_split == -1
    datamodule.prepare_data()
    datamodule.setup(None)

    expected_total = 1281159
    assert (
        datamodule.num_samples
        == expected_total - datamodule.num_classes * datamodule.num_images_per_val_class
    )
    for loader_fn in [
        datamodule.train_dataloader,
        datamodule.val_dataloader,
        datamodule.test_dataloader,
    ]:
        loader = loader_fn()
        for x, y in itertools.islice(loader, 1):
            assert x.shape == (batch_size, 3, 32, 32)
            assert y.shape == (batch_size,)
            break


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    assert SCRATCH
    test_dataset_download_works(SCRATCH / "data")

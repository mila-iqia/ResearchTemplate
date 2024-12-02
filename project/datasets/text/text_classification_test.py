from __future__ import annotations

import huggingface_hub.errors
import lightning
import pytest

from project.datamodules.text.text_classification import TextClassificationDataModule
from project.experiment import instantiate_datamodule
from project.utils.testutils import get_config_loader

datamodule_configs = ["glue_cola"]


@pytest.fixture()
def datamodule(
    request: pytest.FixtureRequest,
) -> lightning.LightningDataModule:
    """Fixture that creates the datamodule for the given config."""
    # Load only the datamodule? (assuming it doesn't depend on the network or anything else...)
    from hydra.types import RunMode

    datamodule_config_name = request.param
    # need to pass a datamodule config via indirect parametrization.
    assert isinstance(datamodule_config_name, str)

    config = get_config_loader().load_configuration(
        f"datamodule/{datamodule_config_name}.yaml",
        overrides=[],
        run_mode=RunMode.RUN,
    )
    datamodule_config = config["datamodule"]
    datamodule = instantiate_datamodule(datamodule_config)
    assert datamodule is not None
    return datamodule


@pytest.fixture()
def prepared_datamodule(
    datamodule: TextClassificationDataModule,
    tmp_path_factory: pytest.TempPathFactory,
):
    tmp_path = tmp_path_factory.mktemp("data")
    _scratch_dir = tmp_path / "_scratch"
    _scratch_dir.mkdir()

    _slurm_tmpdir = tmp_path / "_slurm_tmpdir"
    _slurm_tmpdir.mkdir()

    _scratch_before = datamodule.data_dir
    _slurm_tmpdir_before = datamodule.working_path

    datamodule.data_dir = _scratch_dir / f"{datamodule.task_name}_dataset"
    datamodule.working_path = _slurm_tmpdir / f"{datamodule.task_name}_tmp"
    datamodule.processed_dataset_path = (
        datamodule.data_dir / f"{datamodule.hf_dataset_path}_{datamodule.task_name}_dataset"
    )

    datamodule.prepare_data()
    yield datamodule
    # Restore the original value:
    datamodule.data_dir = _scratch_before
    datamodule.working_path = _slurm_tmpdir_before


@pytest.mark.xfail(
    raises=huggingface_hub.errors.HfHubHTTPError,
    strict=False,
    reason="Can sometimes get 'Too many requests for url'",
)
@pytest.mark.parametrize(datamodule.__name__, datamodule_configs, indirect=True)
def test_dataset_location(
    prepared_datamodule: TextClassificationDataModule,
):
    """Test that the dataset is downloaded to the correct location."""
    datamodule = prepared_datamodule
    assert (
        datamodule.working_path.exists()
    ), f"Dataset path {datamodule.working_path} does not exist."

    expected_files = ["dataset_dict.json"]

    for file_name in expected_files:
        file_path = datamodule.working_path / file_name
        assert file_path.exists(), f"Expected file: {file_name} not found at {file_path}."

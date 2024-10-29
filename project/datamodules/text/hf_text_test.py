from __future__ import annotations

import pytest
from omegaconf import DictConfig

from project.datamodules.text.hf_text import HFDataModule
from project.experiment import (
    instantiate_datamodule,
)
from project.utils.hydra_config_utils import get_config_loader
from project.utils.testutils import (
    run_for_all_configs_of_type,
)
from project.utils.typing_utils.protocols import DataModule


@pytest.fixture()
def datamodule(
    datamodule_config: str | None,
    command_line_overrides: list[str] | None,
) -> DataModule:
    """Fixture that creates the datamodule for the given config."""
    # Load only the datamodule? (assuming it doesn't depend on the network or anything else...)
    from hydra.types import RunMode

    config = get_config_loader().load_configuration(
        f"datamodule/{datamodule_config}.yaml",
        overrides=command_line_overrides or [],
        run_mode=RunMode.RUN,
    )
    datamodule_config = config["datamodule"]
    assert isinstance(datamodule_config, DictConfig)
    datamodule = instantiate_datamodule(datamodule_config)
    return datamodule

    # NOTE: creating the datamodule by itself instead of with everything else.


@pytest.fixture()
def prepared_datamodule(
    datamodule: HFDataModule,
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


@run_for_all_configs_of_type("datamodule", HFDataModule)
def test_dataset_location(
    prepared_datamodule: HFDataModule,
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


@run_for_all_configs_of_type("datamodule", HFDataModule)
@pytest.mark.skip(reason="Not implemented")
def test_pretrained_weight_location(
    prepared_datamodule: HFDataModule,
):
    """Test that the pretrained weights are downloaded to the correct location."""
    # datamodule = prepared_datamodule
    pass


## mismatched tasks
# datamodule = HFDataModule(
#    tokenizer="EleutherAI/gpt-neo-125M",
#    hf_dataset_path="roneneldan/TinyStories",
#    dataset_path=SLURM_TMPDIR,
# )

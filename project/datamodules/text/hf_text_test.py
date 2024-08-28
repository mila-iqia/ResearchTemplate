import pytest

from project.datamodules.text.hf_text import HFDataModule
from project.utils.testutils import run_for_all_configs_of_type


@pytest.fixture(scope="session")
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

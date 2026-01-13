import abc
import sys

import hydra_zen
import omegaconf
import pytest
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture
from torch.utils.data import DataLoader

from project.algorithms.lightning_module_tests import convert_list_and_tuples_to_dicts
from project.conftest import algorithm_config
from project.utils.testutils import IN_GITHUB_CLOUD_CI
from project.utils.typing_utils.protocols import DataModule

# Use a dummy, empty algorithm, to keep the datamodule tests independent of the algorithms.
# This is a unit test for the datamodule, so we don't want to involve the algorithm here.


# NOTE: Here these parametrizations are actually inherited by all the test classes that inherit from DataModuleTests
# and run tests for different datamodules. This is a bit unusual, but avoids us having to repeat these parametrizations
# in every single datamodule test class.


@pytest.mark.skipif(
    IN_GITHUB_CLOUD_CI and sys.platform == "darwin",
    reason="Getting weird bugs with MacOS on GitHub CI.",
)
@pytest.mark.parametrize(algorithm_config.__name__, ["no_op"], indirect=True, ids=[""])
class DataModuleTests[DataModuleType: DataModule](abc.ABC):
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

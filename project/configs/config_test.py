"""TODO: Add tests for the configurations?"""

import hydra_zen
import lightning
import omegaconf
import pytest
from hydra.core.config_store import ConfigStore

from project.configs.config import Config
from project.conftest import command_line_overrides
from project.experiment import Experiment, setup_experiment
from project.main import PROJECT_NAME
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID

CONFIG_DIR = REPO_ROOTDIR / PROJECT_NAME / "configs"

experiment_configs = list((CONFIG_DIR / "experiment").glob("*.yaml"))


@pytest.mark.parametrize(
    command_line_overrides.__name__,
    [
        pytest.param(
            f"experiment={experiment.name}",
            marks=pytest.mark.xfail(
                "cluster" in experiment.name and SLURM_JOB_ID is None,
                reason="Needs to be run on a cluster.",
                raises=omegaconf.errors.InterpolationResolutionError,
                strict=True,
            ),
        )
        for experiment in list(experiment_configs)
    ],
    indirect=True,
    ids=[experiment.name for experiment in list(experiment_configs)],
)
def test_can_load_experiment_configs(experiment_config: Config):
    experiment = setup_experiment(experiment_config)
    assert isinstance(experiment, Experiment)


class DummyModule(lightning.LightningModule):
    def __init__(self, bob: int = 123):
        super().__init__()

    def forward(self, x):
        return self.network(x)


@pytest.fixture(scope="session")
def cs():
    state_before = ConfigStore.get_state()
    yield ConfigStore.instance()
    ConfigStore.set_state(state_before)


@pytest.fixture(scope="session")
def register_dummy_configs(cs: ConfigStore):
    cs.store(
        "dummy",
        node=hydra_zen.builds(
            DummyModule,
            zen_partial=False,
            populate_full_signature=True,
        ),
        group="algorithm",
    )
    cs.store(
        "dummy_partial",
        node=hydra_zen.builds(
            DummyModule,
            zen_partial=True,
            populate_full_signature=True,
        ),
        group="algorithm",
    )


@pytest.mark.parametrize(
    "algorithm_config",
    ["dummy", "dummy_partial"],
    indirect=True,
    # scope="module",
)
def test_can_use_algo_without_datamodule(
    register_dummy_configs: None, algorithm: lightning.LightningModule
):
    """Test that we can use an algorithm without a datamodule."""
    assert isinstance(algorithm, DummyModule)

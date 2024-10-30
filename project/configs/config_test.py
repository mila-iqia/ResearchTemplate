"""TODO: Add tests for the configurations?"""

import copy
from unittest.mock import Mock

import hydra_zen
import lightning
import omegaconf
import pytest
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import project
import project.main
from project.conftest import algorithm_config, command_line_overrides
from project.main import PROJECT_NAME
from project.utils.env_vars import REPO_ROOTDIR, SLURM_JOB_ID

CONFIG_DIR = REPO_ROOTDIR / PROJECT_NAME / "configs"

experiment_configs = list((CONFIG_DIR / "experiment").glob("*.yaml"))


@pytest.fixture
def mock_train(monkeypatch: pytest.MonkeyPatch):
    mock_train_fn = Mock(spec=project.main.train)
    monkeypatch.setattr(project.main, project.main.train.__name__, mock_train_fn)
    return mock_train_fn


@pytest.fixture
def mock_evaluate(monkeypatch: pytest.MonkeyPatch):
    mock_eval_fn = Mock(spec=project.main.evaluation, return_value=("fake", 0.0, {}))
    monkeypatch.setattr(project.main, project.main.evaluation.__name__, mock_eval_fn)
    return mock_eval_fn

# The problem is that not all experiment configs
# are to be used in the same way. For example,
# the cluster_sweep_config.yaml needs an
# additional `cluster` argument. Also, the
# example config uses wandb by default, which is
# probably bad, since it might be creating empty
# jobs in wandb during tests (since the logger is
# instantiated in main, even if the train fn is
# mocked.

@pytest.mark.skip(reason="TODO: test is too general")
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
def test_can_load_experiment_configs(
    experiment_dictconfig: DictConfig, mock_train: Mock, mock_evaluate: Mock
):
    # Mock out some part of the `main` function to not actually run anything.

    results = project.main.main(copy.deepcopy(experiment_dictconfig))
    assert results is not None
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()


class DummyModule(lightning.LightningModule):
    def __init__(self, bob: int = 123):
        super().__init__()

    def forward(self, x):
        return self.network(x)


@pytest.fixture()
def cs():
    state_before = ConfigStore.get_state()
    yield ConfigStore.instance()
    ConfigStore.set_state(state_before)


@pytest.fixture()
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


@pytest.mark.skip(
    # we can already load the jax rl example, which does not use a datamodule."
    reason="Broken, but also kind-of redundant."
)
@pytest.mark.parametrize(
    algorithm_config.__name__,
    ["dummy", "dummy_partial"],
    indirect=True,
)
def test_can_use_algo_that_doesnt_use_a_datamodule(
    register_dummy_configs: None, algorithm: lightning.LightningModule
):
    """Test that we can use an algorithm without a datamodule."""
    assert isinstance(algorithm, DummyModule)

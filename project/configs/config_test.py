"""TODO: Add tests for the configurations?"""

from unittest.mock import Mock

import hydra_zen
import lightning
import pytest
from hydra.core.config_store import ConfigStore

import project
import project.main
from project.conftest import algorithm_config
from project.main import PROJECT_NAME
from project.utils.env_vars import REPO_ROOTDIR

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

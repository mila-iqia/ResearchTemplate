"""TODO: Add tests for the configurations?"""

import hydra_zen
import lightning
import pytest
from hydra.core.config_store import ConfigStore

from project.conftest import algorithm_config


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

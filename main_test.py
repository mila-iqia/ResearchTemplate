# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import typing
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf, open_dict

from project.algorithms import Algorithm, Backprop
from project.configs.config import Config
from project.networks import FcNet

if typing.TYPE_CHECKING:
    pass

TEST_SEED = 123


@pytest.fixture
def testing_overrides():
    """Fixture that gives normal command-line overrides to use during unit testing."""
    return [
        f"seed={TEST_SEED}",
        "trainer=debug",
    ]


@pytest.fixture(autouse=True, scope="session")
def set_testing_hydra_dir():
    """TODO: Set the hydra configuration for unit testing, so temporary directories are used.

    NOTE: Might be a good idea to look in `hydra.test_utils` for something useful, e.g.
    `from hydra.test_utils.test_utils import integration_test`
    """


@pytest.fixture(autouse=True)
def setup_hydra(tmp_path: Path):
    with initialize_config_module("project.configs", job_name="test", version_base="1.1"):
        config = compose(config_name="config", return_hydra_config=True)

        # BUG: Weird errors with Hydra variable interpolation.. Setting these manually seems to fix
        # it for now..
        from hydra.conf import HydraHelpConf
        from hydra.core.hydra_config import HydraConfig

        with open_dict(config):
            # BUG: Getting some weird Hydra omegaconf error in unit tests:
            # "MissingMandatoryValue while resolving interpolation: Missing mandatory value:
            # hydra.job.num"
            config.hydra.job.num = 0
            config.hydra.hydra_help = HydraHelpConf(hydra_help="", template="")
            config.hydra.job.id = 0
            config.hydra.runtime.output_dir = str(tmp_path)

        HydraConfig.instance().set_config(config)

        yield

        # HydraConfig.instance().set_config(base_config.conf)


def test_defaults() -> None:
    # config is relative to a module
    config = compose(config_name="config")
    options = OmegaConf.to_object(config)
    assert isinstance(options, Config)

    # TODO: Comparing `functools.partial`s doesn't work?
    # assert options.algorithm == Backprop.HParams()
    assert isinstance(options.algorithm, Backprop.HParams)
    assert False, options.network

    # NOTE: Not doing this since there are lots of paths that vary between a local machine and the
    # Mila cluster.
    # file_regression.check(OmegaConf.to_yaml(config), extension=".yaml")


def _ids(v):
    if isinstance(v, list):
        return ",".join(map(str, v))
    return None


@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        (["algorithm=backprop"], Backprop.HParams),
    ],
    ids=_ids,
)
def test_setting_algorithm(
    overrides: list[str],
    expected_type: type[Algorithm.HParams],
    testing_overrides: list[str],
    tmp_path: Path,
) -> None:
    config = compose(config_name="config", overrides=testing_overrides + overrides)
    assert config.seed == TEST_SEED  # note: from the testing_overrides above.
    options = OmegaConf.to_object(config)
    assert isinstance(options, Config)
    assert isinstance(options.algorithm, expected_type)


@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        (["algorithm=backprop", "network=fcnet"], FcNet.HParams),
    ],
    ids=_ids,
)
def test_setting_network(
    overrides: list[str], expected_type: type[Algorithm.HParams], testing_overrides: list[str]
) -> None:
    config = compose(config_name="config", overrides=testing_overrides + overrides)
    options = OmegaConf.to_object(config)
    assert isinstance(options, Config)
    assert isinstance(options.network, expected_type)


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!

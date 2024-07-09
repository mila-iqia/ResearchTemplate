# ADAPTED FROM https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
from __future__ import annotations

import typing
from pathlib import Path

import hydra_zen
import pytest

from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.conftest import setup_hydra_for_tests_and_compose, use_overrides
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.networks.fcnet import FcNet
from project.utils.hydra_utils import resolve_dictconfig

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


@use_overrides([""])
def test_defaults(experiment_config: Config) -> None:
    assert isinstance(experiment_config.algorithm, ExampleAlgorithm.HParams)
    assert (
        isinstance(experiment_config.datamodule, CIFAR10DataModule)
        or hydra_zen.get_target(experiment_config.datamodule) is CIFAR10DataModule
    )


def _ids(v):
    if isinstance(v, list):
        return ",".join(map(str, v))
    return None


@pytest.mark.xfail(raises=ValueError, reason="TODO: Hydra config is already set")
@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        (["algorithm=example_algo"], ExampleAlgorithm.HParams),
    ],
    ids=_ids,
)
def test_setting_algorithm(
    overrides: list[str],
    expected_type: type,
    testing_overrides: list[str],
    tmp_path: Path,
) -> None:
    with setup_hydra_for_tests_and_compose(
        all_overrides=testing_overrides + overrides, tmp_path=tmp_path
    ) as dictconfig:
        assert dictconfig.seed == TEST_SEED  # note: from the testing_overrides above.
        config = resolve_dictconfig(dictconfig)
        assert isinstance(config, Config)
        assert isinstance(config.algorithm, expected_type)


@pytest.mark.xfail(raises=ValueError, reason="TODO: Hydra config is already set")
@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        (["algorithm=example_algo", "network=fcnet"], FcNet),
    ],
    ids=_ids,
)
def test_setting_network(
    overrides: list[str],
    expected_type: type,
    testing_overrides: list[str],
    tmp_path: Path,
) -> None:
    # NOTE: Still unclear on the difference between initialize and initialize_config_module
    with setup_hydra_for_tests_and_compose(
        all_overrides=testing_overrides + overrides, tmp_path=tmp_path
    ) as dictconfig:
        options = resolve_dictconfig(dictconfig)
    assert isinstance(options, Config)
    assert hydra_zen.get_target(options.network) is expected_type


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!

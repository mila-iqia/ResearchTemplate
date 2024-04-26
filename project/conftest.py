from __future__ import annotations

import os
import random
import sys
import typing
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import seed_everything
from omegaconf import DictConfig, open_dict
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.algorithms.common.hooks import (
    save_input_output_shapes,
)
from project.configs.config import Config
from project.configs.datamodule import DATA_DIR
from project.datamodules.bases.vision import VisionDataModule, num_cpus_on_node
from project.datamodules.image_classification import ImageClassificationDataModule
from project.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    instantiate_network,
    instantiate_trainer,
    seed_rng,
    setup_experiment,
    setup_logging,
)
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.tensor_regression import TensorRegressionFixture
from project.utils.testutils import default_marks_for_config_name
from project.utils.types import is_sequence_of
from project.utils.types.protocols import DataModule

if typing.TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet
    from _pytest.python import Function

    Param = str | tuple[str, ...] | ParameterSet


logger = get_logger(__name__)

DEFAULT_TIMEOUT = 1.0

# fast = pytest.mark.fast
# fast = pytest.mark.fast
# fast_after_fixtures = pytest.mark.timeout(DEFAULT_TIMEOUT, func_only=True)
DEFAULT_SEED = 42


def pytest_collection_modifyitems(config: pytest.Config, items: list[Function]):
    # NOTE: One can select multiple marks like so: `pytest -m "fast or very_fast"`
    cutoff_time: float | None = config.getoption("--shorter-than", default=None)  # type: ignore

    if cutoff_time is not None:
        if config.getoption("--slow"):
            raise RuntimeError(
                "Can't use both --shorter-than (a cutoff time) and --slow (also run slow tests) since slow tests have no cutoff time!"
            )

    # This -m flag could also be something more complicated like 'fast and not slow', but
    # keeping it simple for now.
    only_running_slow_tests = config.getoption("-m", default=None) == "slow"  # type: ignore

    very_fast_time = DEFAULT_TIMEOUT / 10
    very_fast_timeout_mark = pytest.mark.timeout(very_fast_time, func_only=False)

    fast_time = DEFAULT_TIMEOUT
    # NOTE: The setup time doesn't seem to be properly included in the timeout.
    fast_timeout = pytest.mark.timeout(fast_time, func_only=True)

    indices_to_remove: list[int] = []
    for _node_index, node in enumerate(items):
        # timeout value of the test. None for unknown length.
        test_timeout: float | None = None
        if node.get_closest_marker("very_fast"):
            test_timeout = very_fast_time
            node.add_marker(very_fast_timeout_mark)
        elif node.get_closest_marker("fast"):
            test_timeout = fast_time
            node.add_marker(fast_timeout)
        elif timeout_marker := node.get_closest_marker("timeout"):
            test_timeout = timeout_marker.args[0]
            assert isinstance(test_timeout, int | float)
        elif node.get_closest_marker("slow"):
            # pytest-skip-slow already handles this.
            test_timeout = None
            running_slow_tests = config.getoption("--slow")
            if not running_slow_tests:
                indices_to_remove.append(_node_index)
                continue
        else:
            logger.debug(
                f"Test {node.name} doesn't have a `fast`, `very_fast`, `slow` or `timeout` mark. "
                "Assuming it's fast to run (after test setup)."
            )
            node.add_marker(fast_timeout)
            test_timeout = fast_time

        if cutoff_time is not None:
            assert cutoff_time > 0
            if test_timeout is not None:
                assert test_timeout > 0
                if test_timeout > cutoff_time:
                    node.add_marker(
                        pytest.mark.skip(f"Test takes longer than {cutoff_time}s to run.")
                    )
                    # Note: could also remove indices so we don't have thousands of skipped tests..
                    indices_to_remove.append(_node_index)
                    continue
        elif only_running_slow_tests:
            # IDEA: If we do pytest -m slow --slow, we'd also want to include tests that have a
            # long(er) timeout than ...?
            pass

    if indices_to_remove:
        removed = len(indices_to_remove)
        total = len(items)
        warnings.warn(
            RuntimeWarning(
                f"De-selecting {removed/total:.0%} of tests ({removed}/{total}) because of their length."
            )
        )

    for index in sorted(indices_to_remove, reverse=True):
        items.pop(index)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(autouse=True)
def seed(request: pytest.FixtureRequest):
    """Fixture that seeds everything for reproducibility and yields the random seed used."""
    random_seed = getattr(request, "param", DEFAULT_SEED)
    assert isinstance(random_seed, int) or random_seed is None

    random_state = random.getstate()
    np_random_state = np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        seed_everything(random_seed, workers=True)
        yield random_seed

    random.setstate(random_state)
    np.random.set_state(np_random_state)


@pytest.fixture(scope="session")
def accelerator(request: pytest.FixtureRequest):
    """Returns the accelerator to use during unit tests.

    By default, if cuda is available, returns "cuda". If the tests are run with -vvv, then also
    runs CPU.
    """
    default_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    accelerator: str = getattr(request, "param", default_accelerator)

    if accelerator == "gpu" and not torch.cuda.is_available():
        pytest.skip(reason="GPU not available")
    if accelerator == "cpu" and torch.cuda.is_available():
        if "-vvv" not in sys.argv:
            pytest.skip(
                reason=(
                    "GPU is available and this would take a while on CPU."
                    "Only runs when -vvv is passed."
                ),
            )

    return accelerator


_cuda_available = torch.cuda.is_available()


@pytest.fixture(
    scope="session",
    params=None,
    # ids=lambda args: f"gpus={args}" if _cuda_available else f"cpus={args}",
)
def num_devices_to_use(accelerator: str, request: pytest.FixtureRequest) -> int:
    if accelerator == "gpu":
        num_gpus = getattr(request, "param", 1)
        assert isinstance(num_gpus, int)
        return num_gpus  # Use only one GPU by default.
    else:
        assert accelerator == "cpu"
        return request.param


def run_with_multiple_devices(test_fn: Callable) -> pytest.MarkDecorator:
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        return pytest.mark.parametrize(
            num_devices_to_use.__name__,
            list(range(1, gpus + 1)),
            indirect=True,
            ids=[f"gpus={i}" for i in range(1, gpus + 1)],
        )
    return pytest.mark.parametrize(
        num_devices_to_use.__name__,
        [num_cpus_on_node()],
        indirect=True,
        ids=[""],
    )(test_fn)


@pytest.fixture(scope="session")
def device(accelerator: str) -> torch.device:
    worker_index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    if accelerator == "gpu":
        return torch.device(f"cuda:{worker_index % torch.cuda.device_count()}")
    if accelerator == "cpu":
        return torch.device("cpu")
    raise NotImplementedError(accelerator)


@pytest.fixture(scope="session")
def devices(accelerator: str, num_devices_to_use: int) -> list[int] | int:
    """Fixture that creates the 'devices' argument for the Trainer config."""
    _worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
    worker_index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    assert accelerator in ["cpu", "gpu"]
    if accelerator == "cpu":
        n_cpus = os.cpu_count() or torch.get_num_threads()
        num_devices_to_use = min(num_devices_to_use, n_cpus)
        logger.info(f"Using {num_devices_to_use} CPUs.")
        # NOTE: PyTorch-Lightning Trainer expects to get a number of CPUs, not a list of CPU ids.
        return num_devices_to_use
    if accelerator == "gpu":
        n_gpus = torch.cuda.device_count()
        first_gpu_to_use = worker_index % n_gpus
        num_devices_to_use = min(num_devices_to_use, n_gpus)
        gpus_to_use = sorted(
            set(((np.arange(num_devices_to_use) + first_gpu_to_use) % n_gpus).tolist())
        )
        logger.info(f"Using GPUs: {gpus_to_use}")
        return gpus_to_use
    return 1  # Use only one GPU by default if not distributed.


def _override_param_id(override: Param) -> str:
    if not override:
        return ""
    if isinstance(override, str):
        override = (override,)
    if is_sequence_of(override, str):
        return " ".join(override)
    return str(override)


@pytest.fixture(
    scope="session",
    ids=_override_param_id,
)
def overrides(request: pytest.FixtureRequest):
    """Fixture that makes it possible to specify command-line overrides to use in a given test.

    Tests that require running an experiment should use the `experiment_config` fixture below.

    Multiple test using the same overrides will use the same experiment.
    """
    cmdline_overrides = getattr(request, "param", ())
    assert isinstance(cmdline_overrides, str | list | tuple)
    if isinstance(cmdline_overrides, str):
        cmdline_overrides = cmdline_overrides.split()
    cmdline_overrides = tuple(cmdline_overrides)
    assert all(isinstance(override, str) for override in cmdline_overrides)
    return cmdline_overrides


def use_overrides(command_line_overrides: Param | list[Param], ids=None):
    """Marks a test so that it can use components created using the given command-line arguments.

    For example:

    ```python
    @use_overrides("algorithm=my_algo network=fcnet")
    def test_my_algo(algorithm: MyAlgorithm):
        #The algorithm will be setup the same as if we did
        #   `python main.py algorithm=my_algo network=fcnet`.
        ...
    ```
    """
    return pytest.mark.parametrize(
        overrides.__name__,
        (
            [command_line_overrides]
            if isinstance(command_line_overrides, str | tuple)
            else command_line_overrides
        ),
        indirect=True,
        ids=ids if ids is not None else _override_param_id,
    )


@contextmanager
def setup_hydra_for_tests_and_compose(
    all_overrides: list[str] | None,
    tmp_path_factory: pytest.TempPathFactory | None = None,
    tmp_path: Path | None = None,
):
    with initialize_config_module(
        config_module="project.configs",
        job_name="test",
        version_base="1.2",
    ):
        config = compose(
            config_name="config",
            overrides=all_overrides,
            return_hydra_config=True,
        )

        # BUG: Weird errors with Hydra variable interpolation.. Setting these manually seems
        # to fix it for now..
        from hydra.conf import HydraHelpConf
        from hydra.core.hydra_config import HydraConfig

        with open_dict(config):
            # BUG: Getting some weird Hydra omegaconf error in unit tests:
            # "MissingMandatoryValue while resolving interpolation: Missing mandatory value:
            # hydra.job.num"
            config.hydra.job.num = 0
            config.hydra.hydra_help = HydraHelpConf(hydra_help="", template="")
            config.hydra.job.id = 0
            config.hydra.runtime.output_dir = str(
                tmp_path_factory.mktemp(basename="output", numbered=True)
                if tmp_path_factory
                else tmp_path
            )
        HydraConfig.instance().set_config(config)
        yield config


def _add_default_marks_for_config_name(config_name: str, request: pytest.FixtureRequest):
    """Applies some default marks to tests when running with this config (if any)."""
    if config_name in default_marks_for_config_name:
        for marker in default_marks_for_config_name[config_name]:
            request.applymarker(marker)


@pytest.fixture(scope="session")
def algorithm_name(request: pytest.FixtureRequest) -> str | None:
    algorithm_config_name = getattr(request, "param", None)
    if algorithm_config_name:
        _add_default_marks_for_config_name(algorithm_config_name, request)
    return


@pytest.fixture(scope="session")
def datamodule_name(request: pytest.FixtureRequest) -> str | None:
    datamodule_config_name = getattr(request, "param", None)
    if datamodule_config_name:
        _add_default_marks_for_config_name(datamodule_config_name, request)
    return datamodule_config_name


@pytest.fixture(scope="session")
def network_name(request: pytest.FixtureRequest) -> str | None:
    network_config_name = getattr(request, "param", None)
    if network_config_name:
        _add_default_marks_for_config_name(network_config_name, request)
    return network_config_name


@pytest.fixture(scope="session")
def experiment_dictconfig(
    tmp_path_factory: pytest.TempPathFactory,
    devices: str,
    accelerator: str,
    algorithm_name: str | None,
    datamodule_name: str | None,
    network_name: str | None,
    overrides: tuple[str, ...],
) -> Generator[DictConfig, None, None]:
    tmp_path = tmp_path_factory.mktemp("experiment_testing")

    default_overrides = [
        # NOTE: if we were to run the test in a slurm job, this wouldn't make sense.
        "seed=42",
        f"trainer.devices={devices}",
        f"trainer.accelerator={accelerator}",
    ]
    if not any("trainer.default_root_dir" in override for override in overrides):
        default_overrides.append(f"++trainer.default_root_dir={tmp_path}")
    if algorithm_name:
        default_overrides.append(f"algorithm={algorithm_name}")
    if network_name:
        default_overrides.append(f"network={network_name}")
    if datamodule_name:
        default_overrides.append(f"datamodule={datamodule_name}")

    all_overrides = default_overrides + list(overrides)

    logger.info(
        "This test will run as if this was passed on the command-line:\n"
        + "\n"
        + "```\n"
        + ("python main.py " + " ".join(all_overrides) + "\n")
        + "```\n"
    )

    with setup_hydra_for_tests_and_compose(
        all_overrides=all_overrides,
        tmp_path=tmp_path,
    ) as dict_config:
        yield dict_config


@pytest.fixture(scope="session")
def experiment_config(
    experiment_dictconfig: DictConfig,
) -> Generator[Config, None, None]:
    config = resolve_dictconfig(experiment_dictconfig)
    yield config


@pytest.fixture(scope="module")
def experiment(experiment_config: Config):
    experiment = setup_experiment(experiment_config)
    yield experiment


@pytest.fixture(scope="session")
def common_setup_experiment_part(experiment_config: Config):
    """Fixture that is used to run the common part of `setup_experiment`.

    This is there so that we can instantiate only one or a few of the experiment components (e.g.
    only the Network), while also only doing the common part once if we were to use more than one
    of these components and their associated fixtures below.
    """
    setup_logging(experiment_config)
    seed_rng(experiment_config)


@pytest.fixture(scope="function")
def trainer(experiment_config: Config, common_setup_experiment_part: None) -> pl.Trainer:
    return instantiate_trainer(experiment_config)


@pytest.fixture(scope="session")
def datamodule(experiment_config: Config, common_setup_experiment_part: None) -> DataModule:
    # NOTE: creating the datamodule by itself instead of with everything else.
    return instantiate_datamodule(experiment_config)


@pytest.fixture(scope="session")
def train_dataloader(datamodule: DataModule) -> DataLoader:
    if isinstance(datamodule, VisionDataModule) or hasattr(datamodule, "num_workers"):
        datamodule.num_workers = 0  # type: ignore
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    return train_dataloader


@pytest.fixture(scope="session")
def training_batch(
    train_dataloader: DataLoader, device: torch.device
) -> tuple[Tensor, ...] | dict[str, Tensor]:
    # Get a batch of data from the datamodule so we can initialize any lazy weights in the Network.
    dataloader_iterator = iter(train_dataloader)
    batch = next(dataloader_iterator)
    if is_sequence_of(batch, Tensor):
        batch = tuple(t.to(device=device) for t in batch)
        return batch
    else:
        assert isinstance(batch, dict) and is_sequence_of(batch.values(), Tensor)
        batch = {k: v.to(device=device) for k, v in batch.items()}
        return batch


@pytest.fixture(scope="session")
def x_y(training_batch: tuple[Tensor, ...], datamodule: DataModule) -> tuple[Tensor, Tensor]:
    """Returns a batch of data from the training set of an image classification datamodule."""
    if len(training_batch) != 2 or not isinstance(datamodule, ImageClassificationDataModule):
        pytest.skip(
            reason=(
                f"Test requires a batch of classification data (with 2 elements), batch has "
                f"{len(training_batch)}."
            )
        )
    return training_batch


@pytest.fixture(scope="session")
def num_classes(datamodule: DataModule) -> int:
    """Returns a batch of data from the training set of an image classification datamodule."""
    if not isinstance(datamodule, ImageClassificationDataModule):
        pytest.skip(
            reason=(
                f"Test requires an ImageClassificationDataModule, but datamodule is of type "
                f"{type(datamodule).__name__}"
            )
        )
    return datamodule.num_classes


@pytest.fixture(scope="session")
def input(training_batch: tuple[Tensor, Tensor]) -> Tensor:
    return training_batch[0]


@pytest.fixture(scope="session")
def network(
    experiment_config: Config,
    datamodule: DataModule,
    device: torch.device,
    input: Tensor,
    request: pytest.FixtureRequest,
):
    network = instantiate_network(experiment_config, datamodule=datamodule).to(device)
    try:
        with save_input_output_shapes(network):
            _ = network(input)
    except RuntimeError as err:
        logger.error(f"Error when running the network: {err}")
        request.node.add_marker(
            pytest.mark.xfail(
                raises=RuntimeError,
                reason="Network doesn't seem compatible this dataset.",
            )
        )
    return network


@pytest.fixture(scope="function")
def algorithm(experiment_config: Config, datamodule: DataModule, network: nn.Module):
    return instantiate_algorithm(experiment_config, datamodule=datamodule, network=network)


@pytest.fixture(scope="session")
def classifier_network(network: nn.Module, x_y: tuple[Tensor, Tensor], datamodule: DataModule):
    """Renames the "network" fixture to `classifier_network` if it is indeed an image classifier.

    Skips dependent tests if `network` isn't a classifier.
    """
    with torch.no_grad():
        preds = network(x_y[0])
    if (
        not isinstance(datamodule, ImageClassificationDataModule)
        or not isinstance(preds, Tensor)
        or preds.shape[-1] != datamodule.num_classes
    ):
        pytest.skip(
            reason="The network isn't a classifier or the datamodule isn't an image classification datamodule."
        )
    return network


@pytest.fixture
def make_torch_deterministic():
    """Set torch to deterministic mode for unit tests that use the tensor_regression fixture."""
    mode_before = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode("error")
    yield
    torch.set_deterministic_debug_mode(mode_before)


@pytest.fixture
def tensor_regression(
    datadir: Path,
    original_datadir: Path,
    request: pytest.FixtureRequest,
    ndarrays_regression: NDArraysRegressionFixture,
    data_regression: DataRegressionFixture,
    monkeypatch: pytest.MonkeyPatch,
    make_torch_deterministic: None,
) -> TensorRegressionFixture:
    """Similar to num_regression, but supports numpy arrays with arbitrary shape. The dictionary is
    stored as an NPZ file. The values of the dictionary must be accepted by ``np.asarray``.

    Example::

        def test_some_data(tensor_regression):
            points, values = some_function()
            tensor_regression.check(
                {
                    'points': points,  # tensor with shape (100, 3)
                    'values': values,  # tensor with shape (100,)
                },
                default_tolerance=dict(atol=1e-8, rtol=1e-8)
            )
    """
    return TensorRegressionFixture(
        datadir=datadir,
        original_datadir=original_datadir,
        request=request,
        ndarrays_regression=ndarrays_regression,
        data_regression=data_regression,
        monkeypatch=monkeypatch,
    )


# Incremental testing: https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps
# content of conftest.py


# store history of failures per test class name and per index in parametrize (if parametrize used)
_test_failed_incremental: dict[str, dict[tuple[int, ...], str]] = {}


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values()) if hasattr(item, "callspec") else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values()) if hasattr(item, "callspec") else ()
            )
            # retrieve the name of the first test function to fail for this class name and index
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            # if name found, test has failed for the combination of class name & test name
            if test_name is not None:
                pytest.xfail(f"previous test failed ({test_name})")

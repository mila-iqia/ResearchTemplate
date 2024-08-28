"""Fixtures and test utilities.

This module contains [PyTest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) that are used
by tests.

## How this works

Our goal here is to make sure that the way we create networks/datasets/algorithms during tests match
as closely as possible how they are created normally in a real run.
For example, when running `python project/main.py algorithm=example`.

We achieve this like so: All the components of an experiment are created using fixtures.
The first fixtures to be invoked are the ones that would correspond to command-line arguments.
The fixtures for command-line arguments


For example, one of the fixtures which is created first is [datamodule_config][project.conftest.datamodule_config].

The first fixtures to be created are the [datamodule_config][project.conftest.datamodule_config], `network_config` and `algorithm_config`, along with `overrides`.
From these, the `experiment_dictconfig` is created

```mermaid
---
title: Fixture dependency graph
---
flowchart TD
datamodule_config[
    <a href="#project.conftest.datamodule_config">datamodule_config</a>
] -- 'datamodule=A' --> command_line_arguments
network_config[
    <a href="#project.conftest.network_config">network_config</a>:
] -- 'network=B' --> command_line_arguments
algorithm_config[
    <a href="#project.conftest.algorithm_config">algorithm_config</a>
] -- 'algorithm=C' --> command_line_arguments
overrides[
    <a href="#project.conftest.overrides">overrides</a>
] -- 'seed=123' --> command_line_arguments
command_line_arguments[
    <a href="#project.conftest.command_line_arguments">command_line_arguments</a>
] -- load configs for 'datamodule=A network=B algorithm=C seed=123' --> experiment_dictconfig
experiment_dictconfig[
    <a href="#project.conftest.experiment_dictconfig">experiment_dictconfig</a>
] -- instantiate objects from configs --> experiment_config
experiment_config[
    <a href="#project.conftest.experiment_config">experiment_config</a>
] --> datamodule & network & algorithm
datamodule[
    <a href="#project.conftest.datamodule">datamodule</a>
] --> algorithm
network[
    <a href="#project.conftest.network">network</a>
] --> algorithm
algorithm[
    <a href="#project.conftest.algorithm">algorithm</a>
] -- is used by --> some_test
algorithm & network & datamodule -- is used by --> some_other_test
```
"""

from __future__ import annotations

import os
import sys
import typing
import warnings
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import flax.linen
import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.conf import HydraHelpConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch import Tensor, nn
from torch.utils.data import DataLoader

from project.configs.config import Config
from project.datamodules.vision import VisionDataModule
from project.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    instantiate_network,
    instantiate_trainer,
    seed_rng,
    setup_logging,
)
from project.main import PROJECT_NAME
from project.utils.hydra_config_utils import get_config_loader
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.testutils import (
    PARAM_WHEN_USED_MARK_NAME,
    default_marks_for_config_combinations,
    default_marks_for_config_name,
    seeded_rng,
)
from project.utils.typing_utils import is_mapping_of, is_sequence_of
from project.utils.typing_utils.protocols import DataModule

if typing.TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet
    from _pytest.python import Function

    Param = str | tuple[str, ...] | ParameterSet


logger = get_logger(__name__)

DEFAULT_TIMEOUT = 1.0
DEFAULT_SEED = 42


@pytest.fixture(scope="session")
def algorithm_config(request: pytest.FixtureRequest) -> str | None:
    """The algorithm config to use in the experiment, as if `algorithm=<value>` was passed.

    This is parametrized with all the configurations for a given algorithm type when using the
    included tests, for example as is done in [project.algorithms.example_test][].
    """
    algorithm_config_name = getattr(request, "param", None)
    if algorithm_config_name:
        _add_default_marks_for_config_name(algorithm_config_name, request)
    return algorithm_config_name


@pytest.fixture(scope="session")
def datamodule_config(request: pytest.FixtureRequest) -> str | None:
    """The datamodule config to use in the experiment, as if `datamodule=<value>` was passed."""

    datamodule_config_name = getattr(request, "param", None)
    if datamodule_config_name:
        _add_default_marks_for_config_name(datamodule_config_name, request)
    return datamodule_config_name


@pytest.fixture(scope="session")
def network_config(request: pytest.FixtureRequest) -> str | None:
    """The network config to use in the experiment, as if `network=<value>` was passed."""
    network_config_name = getattr(request, "param", None)
    if network_config_name:
        _add_default_marks_for_config_name(network_config_name, request)
    return network_config_name


@pytest.fixture(scope="session")
def command_line_arguments(
    devices: str,
    accelerator: str,
    algorithm_config: str | None,
    datamodule_config: str | None,
    network_config: str | None,
    overrides: tuple[str, ...],
):
    """Fixture that returns the command-line arguments that will be passed to Hydra to run the
    experiment.

    The `algorithm_config`, `network_config` and `datamodule_config` values here are parametrized
    indirectly by most tests using the [`project.utils.testutils.run_for_all_configs_of_type`][]
    function so that the respective components are created in the same way as they
    would be by Hydra in a regular run.
    """

    combination = set([datamodule_config, network_config, algorithm_config])
    for configs, marks in default_marks_for_config_combinations.items():
        marks = [marks] if not isinstance(marks, list | tuple) else marks
        configs = set(configs)
        if combination >= configs:
            # warnings.warn(f"Applying markers because {combination} contains {configs}")
            # There is a combination of potentially unsupported configs here, e.g. MNIST and ResNets.
            pytest.skip(reason=f"Combination {combination} contains {configs}.")
            # for mark in marks:
            #     request.applymarker(mark)

    default_overrides = [
        # NOTE: if we were to run the test in a slurm job, this wouldn't make sense.
        f"trainer.devices={devices}",
        f"trainer.accelerator={accelerator}",
        # TODO: Setting this here, which actually impacts the tests!
        "seed=42",
    ]
    if algorithm_config:
        default_overrides.append(f"algorithm={algorithm_config}")
    if network_config:
        default_overrides.append(f"network={network_config}")
    if datamodule_config:
        default_overrides.append(f"datamodule={datamodule_config}")

    all_overrides = default_overrides + list(overrides)
    return all_overrides


@pytest.fixture(scope="session")
def experiment_dictconfig(
    command_line_arguments: list[str], tmp_path_factory: pytest.TempPathFactory
) -> DictConfig:
    """The `omegaconf.DictConfig` that is created by Hydra from the command-line arguments.

    Any interpolations in the configs will *not* have been resolved at this point.
    """
    logger.info(
        "This test will run as if this was passed on the command-line:\n"
        + "\n"
        + "```\n"
        + ("python main.py " + " ".join(command_line_arguments) + "\n")
        + "```\n"
    )

    tmp_path = tmp_path_factory.mktemp("test")
    if not any("trainer.default_root_dir" in override for override in command_line_arguments):
        command_line_arguments = command_line_arguments + [
            f"++trainer.default_root_dir={tmp_path}"
        ]

    with _setup_hydra_for_tests_and_compose(
        all_overrides=command_line_arguments,
        tmp_path_factory=tmp_path_factory,
    ) as dict_config:
        return dict_config


@pytest.fixture(scope="session")
def experiment_config(
    experiment_dictconfig: DictConfig,
) -> Config:
    """The experiment configuration, with all interpolations resolved."""
    config = resolve_dictconfig(experiment_dictconfig)
    return config


@pytest.fixture(scope="session")
def network(
    experiment_config: Config,
    datamodule: DataModule,
    device: torch.device,
    training_batch: Any,
):
    with device:
        network = instantiate_network(experiment_config, datamodule=datamodule)

    if isinstance(network, flax.linen.Module):
        return network

    if any(torch.nn.parameter.is_lazy(p) for p in network.parameters()):
        # a bit ugly, but we need to initialize any lazy weights before we pass the network
        # to the tests.
        # TODO: Investigate the false positives with example_from_config, resnets, cifar10
        if isinstance(training_batch, tuple):
            input = training_batch[0]
            _ = network(input)
    return network


@pytest.fixture(scope="session")
def datamodule(
    # experiment_config: Config,
    # _common_setup_experiment_part: None,
    datamodule_config: str | None,
    overrides: list[str] | None,
) -> DataModule:
    """Fixture that creates the datamodule for the given config."""
    if datamodule_config is None:
        datamodule_config = "cifar10"
    # NOTE: creating the datamodule by itself instead of with everything else.
    # Load only the datamodule? (assuming it doesn't depend on the network or anything else...)
    from hydra.types import RunMode

    config = get_config_loader().load_configuration(
        f"datamodule/{datamodule_config}.yaml",
        overrides=overrides or [],
        run_mode=RunMode.RUN,
    )
    datamodule_config = config["datamodule"]
    assert isinstance(datamodule_config, DictConfig)
    datamodule = instantiate_datamodule(datamodule_config)
    return datamodule
    return instantiate_datamodule(experiment_config.datamodule)


@pytest.fixture(scope="function")
def algorithm(experiment_config: Config, datamodule: DataModule, network: nn.Module):
    """Fixture that creates the "algorithm" (a
    [LightningModule][lightning.pytorch.core.module.LightningModule])."""
    return instantiate_algorithm(experiment_config, datamodule=datamodule, network=network)


@pytest.fixture(scope="function")
def trainer(
    experiment_config: Config,
    _common_setup_experiment_part: None,  # noqa
) -> pl.Trainer:
    return instantiate_trainer(experiment_config)


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
        assert is_mapping_of(batch, str, torch.Tensor)
        batch = {k: v.to(device=device) for k, v in batch.items()}
        return batch


# @pytest.fixture(scope="module")
# def experiment(experiment_config: Config) -> Experiment:
#     """Instantiates all the components of an experiment and returns them."""
#     experiment = setup_experiment(experiment_config)
#     return experiment


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
    only_running_slow_tests = "slow" in config.getoption("-m", default="")  # type: ignore
    add_timeout_to_unmarked_tests = False  # todo: Add option for this?

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
        elif add_timeout_to_unmarked_tests:
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


@pytest.fixture(autouse=True)
def seed(request: pytest.FixtureRequest, make_torch_deterministic: None):
    """Fixture that seeds everything for reproducibility and yields the random seed used."""
    random_seed = getattr(request, "param", DEFAULT_SEED)
    assert isinstance(random_seed, int) or random_seed is None

    with seeded_rng(random_seed):
        yield random_seed


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
        return getattr(request, "param", 1)


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
    # todo: Use some parametrize_when_used with some additional arg that says that multiple
    # invocations of this should be appended together instead of added to the list. For example:
    # @use_overrides("algorithm=my_algo network=fcnet")
    # @use_overrides("network=bar")
    # should end up doing
    # ```
    # pytest.mark.parametrize("overrides", ["algorithm=my_algo network=fcnet network=bar"], indirect=True)
    # ```

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
def _setup_hydra_for_tests_and_compose(
    all_overrides: list[str] | None,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Context manager that sets up the Hydra configuration for unit tests."""
    with initialize_config_module(
        config_module=f"{PROJECT_NAME}.configs",
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

        with open_dict(config):
            # BUG: Getting some weird Hydra omegaconf error in unit tests:
            # "MissingMandatoryValue while resolving interpolation: Missing mandatory value:
            # hydra.job.num"
            config.hydra.job.num = 0
            config.hydra.hydra_help = HydraHelpConf(hydra_help="", template="")
            config.hydra.job.id = 0
            config.hydra.runtime.output_dir = str(
                tmp_path_factory.mktemp(basename="output", numbered=True)
            )
        HydraConfig.instance().set_config(config)
        yield config


def _add_default_marks_for_config_name(config_name: str, request: pytest.FixtureRequest):
    """Applies some default marks to tests when running with this config (if any)."""
    if config_name in default_marks_for_config_name:
        for marker in default_marks_for_config_name[config_name]:
            request.applymarker(marker)
    # TODO: ALSO add all the marks for config combinations that contain this config?


@pytest.fixture(scope="session")
def _common_setup_experiment_part(experiment_config: Config):
    """Fixture that is used to run the common part of `setup_experiment`.

    This is there so that we can instantiate only one or a few of the experiment components (e.g.
    only the Network), while also only doing the common part once if we were to use more than one
    of these components in a test.
    """
    setup_logging(experiment_config)
    seed_rng(experiment_config)


@pytest.fixture
def make_torch_deterministic():
    """Set torch to deterministic mode for unit tests that use the tensor_regression fixture."""
    mode_before = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode("error")
    yield
    torch.set_deterministic_debug_mode(mode_before)


# Incremental testing: https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps
# content of conftest.py


# store history of failures per test class name and per index in parametrize (if parametrize used)
_test_failed_incremental: dict[str, dict[tuple[int, ...], str]] = {}


def pytest_runtest_makereport(item, call):
    """Used to setup the `pytest.mark.incremental` mark, as described in [this page](https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps)."""

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
    """Used to setup the `pytest.mark.incremental` mark, as described in [this page](https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps)."""
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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Allows one to define custom parametrization schemes or extensions.

    This is used to implement the `parametrize_when_used` mark, which allows one to parametrize an argument when it is used.

    See
    https://docs.pytest.org/en/7.1.x/how-to/parametrize.html#how-to-parametrize-fixtures-and-test-functions
    """
    # IDEA: Accumulate the parametrizations values from multiple calls, instead of doing like
    # `pytest.mark.parametrize`, which only allows one parametrization.
    args_to_parametrized_values: dict[str, list] = defaultdict(list)
    args_to_be_parametrized_markers: dict[str, list[pytest.Mark]] = defaultdict(list)
    arg_can_be_parametrized: dict[str, bool] = {}
    for marker in metafunc.definition.iter_markers(name=PARAM_WHEN_USED_MARK_NAME):
        assert len(marker.args) == 2
        argnames = marker.args[0]
        argvalues = marker.args[1]
        assert isinstance(argnames, str), argnames

        from _pytest.mark.structures import ParameterSet

        argnames, _parametersets = ParameterSet._for_parametrize(
            argnames,
            argvalues,
            metafunc.function,
            metafunc.config,
            nodeid=metafunc.definition.nodeid,
        )
        from _pytest.outcomes import Failed

        assert len(argnames) == 1
        argname = argnames[0]

        if arg_can_be_parametrized.get(argname):
            args_to_parametrized_values[argname].extend(argvalues)
            args_to_be_parametrized_markers[argname].append(marker)
            continue

        # We don't know if the test uses that argument yet, so we check using the same logic as
        # pytest.mark.parametrize would.
        try:
            metafunc._validate_if_using_arg_names(
                argnames, indirect=marker.kwargs.get("indirect", False)
            )
        except Failed:
            # Test doesn't use that argument, dont parametrize it.
            arg_can_be_parametrized[argname] = False
        else:
            arg_can_be_parametrized[argname] = True
            args_to_parametrized_values[argname].extend(argvalues)
            args_to_be_parametrized_markers[argname].append(marker)

    for arg_name, arg_values in args_to_parametrized_values.items():
        # Test uses that argument, parametrize it.

        # remove duplicates and order the parameters deterministically.
        try:
            arg_values = sorted(set(arg_values), key=str)
        except TypeError:
            pass

        # TODO: unsure what mark to pass here, if there were multiple marks for the same argument..
        marker = args_to_be_parametrized_markers[arg_name][-1]
        indirect = marker.kwargs.get("indirect", False)
        metafunc.parametrize(arg_name, arg_values, indirect=indirect, _param_mark=marker)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--shorter-than",
        action="store",
        type=float,
        default=None,
        help="Skip tests that take longer than this.",
    )


def pytest_ignore_collect(path: str):
    p = Path(path)
    # fixme: Trying to fix doctest issues for project/configs/algorithm/lr_scheduler/__init__.py::project.configs.algorithm.lr_scheduler.StepLRConfig
    if p.name in ["lr_scheduler", "optimizer"] and "configs" in p.parts:
        return True
    return False


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fast: mark test as fast to run (after fixtures are setup)")
    config.addinivalue_line(
        "markers", "very_fast: mark test as very fast to run (including test setup)."
    )

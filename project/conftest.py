"""Fixtures and test utilities.

This module contains [PyTest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) that are used
by tests.

## How this works

Our goal here is to make sure that the way we create networks/datasets/algorithms during tests match
as closely as possible how they are created normally in a real run.
For example, when running `python project/main.py algorithm=image_classifier`.

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
algorithm_config[
    <a href="#project.conftest.algorithm_config">algorithm_config</a>
] -- 'algorithm=B' --> command_line_arguments
command_line_overrides[
    <a href="#project.conftest.command_line_overrides">command_line_overrides</a>
] -- 'seed=123' --> command_line_arguments
command_line_arguments[
    <a href="#project.conftest.command_line_arguments">command_line_arguments</a>
] -- load configs for 'datamodule=A algorithm=B seed=123' --> experiment_dictconfig
experiment_dictconfig[
    <a href="#project.conftest.experiment_dictconfig">experiment_dictconfig</a>
] -- instantiate objects from configs --> experiment_config
experiment_config[
    <a href="#project.conftest.experiment_config">experiment_config</a>
] --> datamodule & algorithm
datamodule[
    <a href="#project.conftest.datamodule">datamodule</a>
] --> algorithm
algorithm[
    <a href="#project.conftest.algorithm">algorithm</a>
] -- is used by --> some_test
algorithm & datamodule -- is used by --> some_other_test
```
"""

from __future__ import annotations

import copy
import functools
import operator
import os
import shlex
import sys
import typing
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal

import hydra.errors
import jax
import lightning
import lightning.pytorch
import lightning.pytorch as pl
import lightning.pytorch.utilities
import pytest
import tensor_regression.stats
import torch
from _pytest.outcomes import Skipped, XFailed
from _pytest.python import Function
from _pytest.runner import CallInfo
from hydra import compose, initialize_config_module
from hydra.conf import HydraHelpConf
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.auto_schema import auto_schema_plugin
from hydra_plugins.auto_schema.auto_schema_plugin import add_schemas_to_all_hydra_configs
from omegaconf import DictConfig, open_dict
from tensor_regression.stats import get_simple_attributes
from tensor_regression.to_array import to_ndarray
from torch import Tensor
from torch.utils.data import DataLoader

from project.configs.config import Config
from project.datamodules.vision import VisionDataModule, num_cpus_on_node
from project.experiment import instantiate_datamodule, instantiate_trainer
from project.main import (
    PROJECT_NAME,
    instantiate_algorithm,
    setup_logging,
)
from project.trainers.jax_trainer import JaxTrainer
from project.utils.env_vars import REPO_ROOTDIR
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.testutils import (
    IN_GITHUB_CI,
    PARAM_WHEN_USED_MARK_NAME,
    default_marks_for_config_combinations,
    default_marks_for_config_name,
)
from project.utils.typing_utils import is_sequence_of

if typing.TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

    Param = str | tuple[str, ...] | ParameterSet


logger = get_logger(__name__)

DEFAULT_TIMEOUT = 1.0
DEFAULT_SEED = 42

# Note: Here we attempt to make this happen only once.
auto_schema_plugin.add_schemas_to_all_hydra_configs = functools.cache(
    add_schemas_to_all_hydra_configs
)


fails_on_macOS_in_CI = pytest.mark.xfail(
    sys.platform == "darwin" and IN_GITHUB_CI,
    raises=(RuntimeError, hydra.errors.InstantiationException),
    reason="Raises 'MPS backend out of memory' error on MacOS in GitHub CI.",
)
skip_on_macOS_in_CI = pytest.mark.skipif(
    sys.platform == "darwin" and IN_GITHUB_CI,
    reason="TODO: Fails for some reason on MacOS in GitHub CI.",
)


@pytest.fixture(autouse=True, scope="session")
def prevent_jax_from_reserving_all_the_vram():
    # note; not using monkeypatch because we want this to be session-scoped.
    @contextmanager
    def change_env(variable_name: str, value: str):
        val_before = os.environ.get(variable_name)
        os.environ[variable_name] = value
        yield
        if val_before is None:
            os.environ.pop(variable_name)
        else:
            os.environ[variable_name] = val_before

    # Set these so that we can use torch and jax during tests on the same GPU (and so that Jax lets
    # go of the VRAM it doesn't need anymore.
    # See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html for more info.
    with (
        change_env("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
        change_env("XLA_PYTHON_CLIENT_ALLOCATOR", "platform"),
    ):
        yield


@pytest.fixture(autouse=True)
def original_datadir(original_datadir: Path):
    """Overwrite the original_datadir fixture value to change where regression files are created.

    By default, they are in a folder next to the source. Here instead we move them to a different
    folder to keep the source code folder as neat as we can.

    TODO: The large regression files (the .npz files containing tensors) could be stored in a cache
    on $SCRATCH and referenced to via a symlink in the test folder. There could be some issues
    though if scratch is cleaned up.
    """
    # `original_datadir` is a fixture provided by the `pytest-datadir` package, its value is set
    # based on the test that is currently being run (and its parameters).
    relative_path_to_regression_file = original_datadir.relative_to(REPO_ROOTDIR)
    regression_files_dir = REPO_ROOTDIR / "tests" / ".regression_files"
    return regression_files_dir / relative_path_to_regression_file


@pytest.fixture(scope="session")
def algorithm_config(request: pytest.FixtureRequest) -> str | None:
    """The algorithm config to use in the experiment, as if `algorithm=<value>` was passed.

    This is parametrized with all the configurations for a given algorithm type when using the
    included tests, for example as is done in [project.algorithms.image_classifier_test][].
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
def algorithm_network_config(request: pytest.FixtureRequest) -> str | None:
    """The network config to use in the experiment, as in `algorithm/network=<value>`."""
    network_config_name = getattr(request, "param", None)
    if network_config_name:
        _add_default_marks_for_config_name(network_config_name, request)
    return network_config_name


@pytest.fixture(scope="session")
def command_line_arguments(
    algorithm_config: str | None,
    datamodule_config: str | None,
    algorithm_network_config: str | None,
    command_line_overrides: tuple[str, ...],
    request: pytest.FixtureRequest,
):
    """Fixture that returns the command-line arguments that will be passed to Hydra to run the
    experiment.

    The `algorithm_config`, `network_config` and `datamodule_config` values here are parametrized
    indirectly by most tests using the [`project.utils.testutils.run_for_all_configs_of_type`][]
    function so that the respective components are created in the same way as they
    would be by Hydra in a regular run.
    """
    if param := getattr(request, "param", None):
        # If we manually overwrite the command-line arguments with indirect parametrization,
        # then ignore the rest of the stuff here and just use the provided command-line args.
        # Split the string into a list of command-line arguments if needed.
        if isinstance(param, str):
            return tuple(shlex.split(param))
        assert isinstance(param, list | tuple)
        return tuple(param)

    combination = set([datamodule_config, algorithm_network_config, algorithm_config])
    for configs, marks in default_marks_for_config_combinations.items():
        marks = [marks] if not isinstance(marks, list | tuple) else marks
        configs = set(configs)
        if combination >= configs:
            # warnings.warn(f"Applying markers because {combination} contains {configs}")
            # There is a combination of potentially unsupported configs here, e.g. MNIST and ResNets.
            # BUG: This is supposed to work, but doesn't for some reason!
            # for mark in marks:
            #     request.applymarker(mark)
            # Skipping the test entirely for now.
            pytest.skip(reason=f"Combination {combination} contains {configs}.")

    default_overrides = [
        # NOTE: if we were to run the test in a slurm job, this wouldn't make sense.
        # f"trainer.devices={devices}",
        # f"trainer.accelerator={accelerator}",
        # TODO: Setting this here, which actually impacts the tests!
        "seed=42",
    ]
    if algorithm_config:
        default_overrides.append(f"algorithm={algorithm_config}")
    if algorithm_network_config:
        default_overrides.append(f"algorithm/network={algorithm_network_config}")
    if datamodule_config:
        default_overrides.append(f"datamodule={datamodule_config}")

    all_overrides = default_overrides + list(command_line_overrides)
    return all_overrides


@pytest.fixture(scope="session")
def experiment_dictconfig(
    command_line_arguments: tuple[str, ...], tmp_path_factory: pytest.TempPathFactory
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
        command_line_arguments = tuple(command_line_arguments) + (
            f"++trainer.default_root_dir={tmp_path}",
        )

    with _setup_hydra_for_tests_and_compose(
        all_overrides=list(command_line_arguments),
        tmp_path_factory=tmp_path_factory,
    ) as dict_config:
        return dict_config


@pytest.fixture(scope="function")
def experiment_config(
    experiment_dictconfig: DictConfig,
) -> Config:
    """The experiment configuration, with all interpolations resolved."""
    config = resolve_dictconfig(copy.deepcopy(experiment_dictconfig))
    return config


# BUG: The network has a default config of `resnet18`, which tries to get the
# num_classes from the datamodule. However, the hf_text datamodule doesn't have that attribute,
# and we load the datamodule using the entire experiment config, so loading the network raises an
# error!
# - instantiate(experiment_config).datamodule
# - instantiate(experiment_dictconfig['datamodule'])


@pytest.fixture(scope="session")
def datamodule(experiment_dictconfig: DictConfig) -> lightning.LightningDataModule | None:
    """Fixture that creates the datamodule for the given config."""
    # NOTE: creating the datamodule by itself instead of with everything else.
    return instantiate_datamodule(experiment_dictconfig["datamodule"])


@pytest.fixture(scope="function")
def algorithm(
    experiment_config: Config,
    datamodule: lightning.LightningDataModule | None,
    trainer: lightning.Trainer | JaxTrainer,
    seed: int,
    device: torch.device,
):
    """Fixture that creates the "algorithm" (a
    [LightningModule][lightning.pytorch.core.module.LightningModule])."""
    algorithm = instantiate_algorithm(experiment_config, datamodule=datamodule)
    if isinstance(trainer, lightning.Trainer) and isinstance(algorithm, lightning.LightningModule):
        with trainer.init_module(), device:
            # A bit hacky, but we have to do this because the lightningmodule isn't associated
            # with a Trainer.
            algorithm._device = device
            algorithm.configure_model()
    return algorithm


@pytest.fixture(scope="function")
def trainer(
    experiment_config: Config,
) -> pl.Trainer | JaxTrainer:
    setup_logging(log_level=experiment_config.log_level)
    # put here to copy what's done in main.py
    lightning.seed_everything(experiment_config.seed, workers=True)
    return instantiate_trainer(experiment_config.trainer)


@pytest.fixture(scope="session")
def train_dataloader(
    datamodule: lightning.LightningDataModule | None, request: pytest.FixtureRequest
) -> DataLoader:
    if isinstance(datamodule, VisionDataModule) or hasattr(datamodule, "num_workers"):
        datamodule.num_workers = 0  # type: ignore
    if datamodule is None:
        raise NotImplementedError(
            "This test is trying to use `train_dataloader` directly or indirectly but the "
            "algorithm that is being tested does not use a datamodule (or the test was not "
            "configured properly)! Consider overwriting this fixture in your test class."
        )
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    return train_dataloader


# todo: Remove (unused).
@pytest.fixture(scope="session")
def training_batch(
    train_dataloader: DataLoader, device: torch.device
) -> tuple[Tensor, ...] | dict[str, Tensor]:
    # Get a batch of data from the dataloader.

    # The batch of data will always be the same because the dataloaders are passed a Generator
    # object in their constructor.
    assert isinstance(train_dataloader, DataLoader)
    dataloader_iterator = iter(train_dataloader)

    with torch.random.fork_rng(list(range(torch.cuda.device_count()))):
        # TODO: This ugliness is because torchvision transforms use the global pytorch RNG!
        torch.random.manual_seed(42)
        batch = next(dataloader_iterator)

    return jax.tree.map(operator.methodcaller("to", device=device), batch)


@pytest.fixture(autouse=True, scope="function")
def seed(request: pytest.FixtureRequest, make_torch_deterministic: None):
    """Fixture that seeds everything for reproducibility and yields the random seed used."""
    random_seed = getattr(request, "param", DEFAULT_SEED)
    assert isinstance(random_seed, int) or random_seed is None

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        lightning.seed_everything(random_seed, workers=True)
        yield random_seed


# TODO: Remove this.
@pytest.fixture(scope="session")
def accelerator(request: pytest.FixtureRequest):
    """Returns the accelerator to use during unit tests.

    By default, if cuda is available, returns "cuda". If the tests are run with -vvv, then also
    runs CPU.
    """
    # TODO: Shouldn't we get this from the experiment config instead?

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


@pytest.fixture(scope="session")
def device(accelerator: str) -> torch.device:
    worker_index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    if accelerator == "gpu":
        return torch.device(f"cuda:{worker_index % torch.cuda.device_count()}")
    if accelerator == "cpu":
        return torch.device("cpu")
    raise NotImplementedError(accelerator)


@pytest.fixture(scope="session")
def devices(
    accelerator: str, request: pytest.FixtureRequest
) -> Generator[list[int] | int | Literal["auto"], None, None]:
    """Fixture that creates the 'devices' argument for the Trainer config.

    Splits up the GPUs between pytest-xdist workers when using distributed testing.
    This isn't currently used in the CI.

    TODO: Design dilemna here: Should we be parametrizing the `devices` command-line override and
    force experiments to run with this value during tests? Or should we be changing things based on
    this value in the config?
    """
    # When using pytest-xdist to distribute tests, each worker will use different devices.

    devices = getattr(request, "param", None)
    if devices is not None:
        # The 'devices' flag was set using indirect parametrization.
        return devices

    num_pytest_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    worker_index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))

    if accelerator == "cpu" or (accelerator == "auto" and not torch.cuda.is_available()):
        n_cpus = num_cpus_on_node()
        # Split the CPUS as evenly as possible (last worker might get less).
        if num_pytest_workers == 1:
            yield "auto"
            return
        n_cpus_for_this_worker = (
            n_cpus // num_pytest_workers
            if worker_index != num_pytest_workers - 1
            else n_cpus - n_cpus // num_pytest_workers * (num_pytest_workers - 1)
        )
        assert 1 <= n_cpus_for_this_worker <= n_cpus
        yield n_cpus_for_this_worker
        return

    if accelerator == "gpu" or (accelerator == "auto" and torch.cuda.is_available()):
        # Alternate GPUS between workers.
        n_gpus = torch.cuda.device_count()
        first_gpu_to_use = worker_index % n_gpus
        logger.info(f"Using GPU #{first_gpu_to_use}")
        devices_before = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(first_gpu_to_use)
        yield [first_gpu_to_use]
        if devices_before is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices_before
        return

    yield 1  # Use only one GPU by default if not distributed.


def _override_param_id(override: Param) -> str:
    if not override:
        return ""
    if isinstance(override, str):
        override = (override,)
    if is_sequence_of(override, str):
        return " ".join(override)
    return str(override)


@pytest.fixture(scope="session", ids=_override_param_id)
def command_line_overrides(request: pytest.FixtureRequest) -> tuple[str, ...]:
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


def pytest_runtest_makereport(item: Function, call: CallInfo):
    """Used to setup the `pytest.mark.incremental` mark, as described in the pytest docs.

    See [this page](https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps)
    """
    if "incremental" not in item.keywords:
        return
    # incremental marker is used
    # NOTE: Modified this part to also take into account the type of exception:
    # - If the test raised a Skipped or XFailed, then we don't consider it as a "failure" and let
    #   the following tests run.
    if call.excinfo is not None and not call.excinfo.errisinstance((Skipped, XFailed)):  # type: ignore
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
        _test_failed_incremental.setdefault(cls_name, {}).setdefault(parametrize_index, test_name)


def pytest_runtest_setup(item: Function):
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


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fast: mark test as fast to run (after fixtures are setup)")
    config.addinivalue_line(
        "markers", "very_fast: mark test as very fast to run (including test setup)."
    )


# TODO: remove these, add this fix to the tensor_regression package instead.
@pytest.fixture(autouse=True)
def _dont_use_tensor_hashes_in_regression_files(monkeypatch: pytest.MonkeyPatch):
    """Temporarily remove the hash of tensors from the regression files."""

    monkeypatch.setattr(
        tensor_regression.fixture,
        tensor_regression.fixture.get_simple_attributes.__name__,  # type: ignore
        _patched_simple_attributes,
    )


def _patched_simple_attributes(v, precision: int | None):
    stats = tensor_regression.stats.get_simple_attributes(v, precision=precision)
    stats.pop("hash", None)
    return stats


@get_simple_attributes.register(tuple)
def _get_tuple_attributes(value: tuple, precision: int | None):
    # This is called to get some simple stats to store in regression files during tests, in
    # particular for tuples (since there isn't already a handler for it in the tensor_regression
    # package.)
    # Note: This information about this output is not very descriptive.
    # not this is called only for the `out.past_key_values` entry in the `CausalLMOutputWithPast`
    # that is returned from the forward pass output.
    num_items_to_include = 5  # only show the stats of some of the items.
    return {
        "length": len(value),
        **{
            f"{i}": get_simple_attributes(item, precision=precision)
            for i, item in enumerate(value[:num_items_to_include])
        },
    }


@to_ndarray.register(list)
@to_ndarray.register(tuple)
def _tuple_to_ndarray(v: tuple | list):
    """Convert a tuple of values to a numpy array to be stored in a regression file."""
    # This could get a bit tricky because the items might not have the same shape and so on.
    # However it seems like the ndarrays_regression fixture (which is what tensor_regression uses
    # under the hood) is not complaining about us returning a list here, so we'll leave it at that
    # for now.
    return {i: to_ndarray(v_i) for i, v_i in enumerate(v)}  # type: ignore

"""Utility functions useful for testing."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import itertools
import os
import random
import typing
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from logging import getLogger as get_logger
from typing import Any, Generic, TypeVar

import lightning
import numpy as np
import pytest
import torch
import torchvision.models
from torch import nn

from project.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.datamodules.vision import VisionDataModule
from project.utils.env_vars import NETWORK_DIR
from project.utils.hydra_config_utils import (
    get_all_configs_in_group,
    get_all_configs_in_group_of_type,
)

logger = get_logger(__name__)

IN_GITHUB_CI = "GITHUB_ACTIONS" in os.environ
IN_SELF_HOSTED_GITHUB_CI = IN_GITHUB_CI and "self-hosted" in os.environ.get("RUNNER_LABELS", "")
PARAM_WHEN_USED_MARK_NAME = "parametrize_when_used"


default_marks_for_config_name: dict[str, list[pytest.MarkDecorator]] = {
    "imagenet32": [pytest.mark.slow],
    "inaturalist": [
        pytest.mark.slow,
        pytest.mark.skipif(
            not (NETWORK_DIR and (NETWORK_DIR / "datasets/inat").exists()),
            # strict=True,
            # raises=hydra.errors.InstantiationException,
            reason="Expects to be run on the Mila cluster for now",
        ),
    ],
    "imagenet": [
        pytest.mark.slow,
        pytest.mark.skipif(
            not (NETWORK_DIR and (NETWORK_DIR / "datasets/imagenet").exists()),
            # strict=True,
            # raises=hydra.errors.InstantiationException,
            reason="Expects to be run on a cluster with the ImageNet dataset.",
        ),
    ],
    "vision": [pytest.mark.skip(reason="Base class, shouldn't be instantiated.")],
}
"""Dict with some default marks for some configs name."""

default_marks_for_config_combinations: dict[tuple[str, ...], list[pytest.MarkDecorator]] = {
    ("imagenet", "fcnet"): [
        pytest.mark.xfail(
            reason="FcNet shouldn't be applied to the ImageNet datamodule. It can lead to nans in the parameters."
        )
    ],
    ("imagenet", "jax_fcnet"): [
        pytest.mark.xfail(
            reason="FcNet shouldn't be applied to the ImageNet datamodule. It can lead to nans in the parameters."
        )
    ],
    ("imagenet", "jax_cnn"): [
        pytest.mark.xfail(
            reason="todo: parameters contain nans when overfitting on one batch? Maybe we're "
            "using too many iterations?"
        )
    ],
    **{
        (resnet_config, mnist_dataset_config): [
            pytest.mark.xfail(
                reason="ResNets don't work with MNIST datasets because the image resolution is too small.",
                raises=RuntimeError,
            )
        ]
        for resnet_config, mnist_dataset_config in itertools.product(
            get_all_configs_in_group_of_type("network", torchvision.models.ResNet),
            get_all_configs_in_group_of_type(
                "datamodule", (MNISTDataModule, FashionMNISTDataModule)
            ),
        )
    },
}


def parametrized_fixture(name: str, values: Sequence, ids=None, **kwargs):
    """Small helper function that creates a parametrized pytest fixture for the given values.

    NOTE: When writing a fixture in a test class, use `ParametrizedFixture` instead.
    """

    @pytest.fixture(name=name, params=values, ids=ids or [f"{name}={v}" for v in values], **kwargs)
    def _parametrized_fixture(request: pytest.FixtureRequest):
        return request.param

    return _parametrized_fixture


T = TypeVar("T")


class ParametrizedFixture(Generic[T]):
    """Small helper function that creates a parametrized pytest fixture for the given values.

    The name of the fixture will be the name that is used for this variable on a class.

    For example:

    ```python

    class TestFoo:
        odd = ParametrizedFixture([True, False])

        def test_something(self, odd: bool):
            '''some kind of test that uses odd'''

        # NOTE: This fixture can also be used by other fixtures:

        @pytest.fixture
        def some_number(self, odd: bool):
            return 1 if odd else 2

        def test_foo(self, some_number: int):
            '''some kind of test that uses some_number'''
    ```
    """

    def __init__(self, values: list[T], name: str | None = None, **fixture_kwargs):
        self.values = values
        self.fixture_kwargs = fixture_kwargs
        self.name = name

    def __set_name__(self, owner: Any, name: str):
        self.name = name

    def __get__(self, obj, objtype=None):
        assert self.name is not None
        fixture_kwargs = self.fixture_kwargs.copy()
        fixture_kwargs.setdefault("ids", [f"{self.name}={v}" for v in self.values])

        @pytest.fixture(name=self.name, params=self.values, **fixture_kwargs)
        def _parametrized_fixture_method(request: pytest.FixtureRequest):
            return request.param

        return _parametrized_fixture_method


def run_for_all_datamodules(
    datamodule_names: list[str] | None = None,
    datamodule_name_to_marks: dict[str, pytest.MarkDecorator | list[pytest.MarkDecorator]]
    | None = None,
):
    """Apply this marker to a test to make it run with all available datasets (datamodules).

    The test should use the `datamodule` fixture, either as an input argument to the test
    function or indirectly by using a fixture that depends on the `datamodule` fixture.

    Parameters
    ----------
    datamodule_names: List of datamodule names to use for tests. \
        By default, lists out the generic datamodules (the datamodules that aren't specific to a
        single algorithm, for example the InfGendatamodules of WakeSleep.)

    datamodule_to_marks: Dictionary from datamodule names to pytest marks (e.g. \
        `pytest.mark.xfail`, `pytest.mark.skip`) to use for that particular datamodule.
    """
    return run_for_all_configs_in_group(
        group_name="datamodule",
        config_name_to_marks=datamodule_name_to_marks,
    )


def run_for_all_vision_datamodules():
    return run_for_all_configs_of_type("datamodule", VisionDataModule)


def run_for_all_configs_of_type(
    config_group: str, config_target_type: type, excluding: type | tuple[type, ...] = ()
):
    """Parametrizes a test to run with all the configs in the given group that have targets which
    are subclasses of the given type.

    For example:

    ```python
    @run_for_all_subclasses_of("network", torch.nn.Module)
    def test_something_about_the_network(network: torch.nn.Module):
        ''' This test will run with all the configs in the 'network' group that produce nn.Modules! '''
    ```
    """
    config_names = get_all_configs_in_group_of_type(
        config_group, config_target_type, include_subclasses=True, excluding=excluding
    )
    config_name_to_marks = {
        name: default_marks_for_config_name.get(name, []) for name in config_names
    }
    return run_for_all_configs_in_group(config_group, config_name_to_marks=config_name_to_marks)


def run_for_all_image_classification_datamodules():
    return run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)


def parametrize_when_used(
    arg_name_or_fixture: str | typing.Callable, values: list, indirect: bool | None = None
) -> pytest.MarkDecorator:
    """Fixture that applies `pytest.mark.parametrize` only when the argument is used (directly or
    indirectly).

    When `pytest.mark.parametrize` is applied to a class, all test methods in that class need to
    use the parametrized argument, otherwise an error is raised. This function exists to work around
    this and allows writing test methods that don't use the parametrized argument.

    For example, this works, but would not be possible with `pytest.mark.parametrize`:

    ```python
    import pytest

    @parametrize_when_used("value", [1, 2, 3])
    class TestFoo:
        def test_foo(self, value):
            ...

        def test_bar(self, value):
            ...

        def test_something_else(self):  # This will cause an error!
            pass
    ```

    Parameters
    ----------
    arg_name_or_fixture: The name of the argument to parametrize, or a fixture to parametrize \
        indirectly.
    values: The values to be used to parametrize the test.

    Returns
    -------
    A `pytest.MarkDecorator` that parametrizes the test with the given values only when the argument
    is used (directly or indirectly) by the test.
    """
    if indirect is None:
        indirect = not isinstance(arg_name_or_fixture, str)
    arg_name = (
        arg_name_or_fixture
        if isinstance(arg_name_or_fixture, str)
        else arg_name_or_fixture.__name__
    )
    mark_fn = getattr(pytest.mark, PARAM_WHEN_USED_MARK_NAME)
    return mark_fn(arg_name, values, indirect=indirect)


def run_for_all_configs_in_group(
    group_name: str,
    config_name_to_marks: Mapping[str, pytest.MarkDecorator | list[pytest.MarkDecorator]]
    | None = None,
):
    """Apply this marker to a test to make it run with all configs in a given group.

    This assumes that a "`group_name`_config" fixture is defined, for example, `algorithm_config`,
    `datamodule_config`, `network_config`. This then does an indirect parametrization of that fixture, so that it
    receives the config name as a parameter and returns it.


    The test wrapped test will uses all config from that group if they are used either as an input
    argument to the test function or if it the input argument to a fixture function.

    Parameters
    ----------
    datamodule_names: List of datamodule names to use for tests. \
        By default, lists out the generic datamodules (the datamodules that aren't specific to a
        single algorithm, for example the InfGendatamodules of WakeSleep.)

    datamodule_to_marks: Dictionary from datamodule names to pytest marks (e.g. \
        `pytest.mark.xfail`, `pytest.mark.skip`) to use for that particular datamodule.
    """
    if config_name_to_marks is None:
        config_name_to_marks = {
            k: default_marks_for_config_name.get(k, [])
            for k in get_all_configs_in_group(group_name)
        }
    # Parametrize the fixture (e.g. datamodule_name) indirectly, which will make it take each group
    # member (e.g. datamodule config name), each with a parameterized mark.
    return parametrize_when_used(
        f"{group_name}_config",
        [
            pytest.param(
                config_name,
                marks=tuple(marks) if isinstance(marks, list) else marks,
                # id=f"{group_name}={config_name}",
            )
            for config_name, marks in config_name_to_marks.items()
        ],
        indirect=True,
    )


def assert_all_params_initialized(module: nn.Module):
    for name, param in module.named_parameters():
        assert not isinstance(param, nn.UninitializedParameter | nn.UninitializedBuffer), name


def assert_no_nans_in_params_or_grads(module: nn.Module):
    for name, param in module.named_parameters():
        assert not torch.isnan(param).any(), name
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), name


@contextlib.contextmanager
def fork_rng():
    """Forks the RNG, so that when you return, the RNG is reset to the state that it was previously
    in."""
    rng_state = RngState.get()
    yield
    rng_state.set()


@contextmanager
def seeded_rng(seed: int = 42):
    """Forks the RNG and seeds the torch, numpy, and random RNGs while inside the block."""
    with fork_rng():
        random_state = RngState.seed(seed)
        yield random_state


def _get_cuda_rng_states():
    return tuple(
        torch.cuda.get_rng_state(torch.device("cuda", index=index))
        for index in range(torch.cuda.device_count())
    )


@dataclasses.dataclass(frozen=True)
class RngState:
    random_state: tuple[Any, ...] = dataclasses.field(default_factory=random.getstate)
    numpy_random_state: dict[str, Any] = dataclasses.field(default_factory=np.random.get_state)

    torch_cpu_rng_state: torch.Tensor = torch.get_rng_state()
    torch_device_rng_states: tuple[torch.Tensor, ...] = dataclasses.field(
        default_factory=_get_cuda_rng_states
    )

    @classmethod
    def get(cls):
        """Gets the state of the random/numpy/torch random number generators.

        Note: do a deepcopy just in case the libraries return the rng state "by reference" and keep
        modifying it.
        """
        return copy.deepcopy(cls())

    def set(self):
        """Resets the state of the random/numpy/torch random number generators with the contents of
        `self`."""
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.set_rng_state(self.torch_cpu_rng_state)
        for index, state in enumerate(self.torch_device_rng_states):
            torch.cuda.set_rng_state(state, torch.device("cuda", index=index))

    @classmethod
    def seed(cls, base_seed: int):
        lightning.seed_everything(base_seed, workers=True)
        # random.seed(base_seed)
        # np.random.seed(base_seed)
        # torch.random.manual_seed(base_seed)
        return cls()

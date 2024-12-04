"""Utility functions useful for testing."""

from __future__ import annotations

import functools
import inspect
import itertools
import os
import typing
from collections.abc import Callable, Mapping
from logging import getLogger as get_logger

import hydra
import hydra_zen
import pytest
import torch
import torchvision.models
from hydra.core.config_store import ConfigStore

from project.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from project.datamodules.image_classification.mnist import MNISTDataModule
from project.utils.env_vars import NETWORK_DIR, SLURM_JOB_ID
from project.utils.hydra_utils import get_outer_class

logger = get_logger(__name__)

IN_GITHUB_CI = "GITHUB_ACTIONS" in os.environ
IN_SELF_HOSTED_GITHUB_CI = IN_GITHUB_CI and (
    "self-hosted" in os.environ.get("RUNNER_LABELS", "")
    or (torch.cuda.is_available() and SLURM_JOB_ID is None)
)
IN_GITHUB_CLOUD_CI = IN_GITHUB_CI and not IN_SELF_HOSTED_GITHUB_CI
PARAM_WHEN_USED_MARK_NAME = "parametrize_when_used"


default_marks_for_config_name: dict[str, list[pytest.MarkDecorator]] = {
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


# Doing this once only because it might be a bit expensive.
@functools.cache
def get_config_loader():
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra._internal.utils import create_automatic_config_search_path

    from project.main import PROJECT_NAME

    # TODO: This (loading a config) is actually taking a long time, in part because this is
    # triggering the hydra-auto-schema plugin to add schemas to all the yaml files.
    AutoSchemaPlugin = None
    backup = None
    try:
        from hydra_plugins.auto_schema.auto_schema_plugin import (
            AutoSchemaPlugin,
        )

        backup = AutoSchemaPlugin._ALREADY_DID
        AutoSchemaPlugin._ALREADY_DID = True
    except ImportError:
        pass
    search_path = create_automatic_config_search_path(
        calling_file=None, calling_module=None, config_path=f"pkg://{PROJECT_NAME}.configs"
    )
    if AutoSchemaPlugin is not None:
        AutoSchemaPlugin._ALREADY_DID = backup
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    return config_loader


def get_target_of_config(
    config_group: str, config_name: str, _cs: ConfigStore | None = None
) -> Callable:
    """Returns the class that is to be instantiated by the given config name.

    In the case of inner dataclasses (e.g. Model.HParams), this returns the outer class (Model).
    """
    # TODO: Rework, use the same mechanism as in auto-schema.py
    if _cs is None:
        from project.configs import cs as _cs

    config_loader = get_config_loader()
    _, caching_repo = config_loader._parse_overrides_and_create_caching_repo(
        config_name=None, overrides=[]
    )
    # todo: support both `.yml` and `.yaml` extensions for config files.
    for extension in ["yaml", "yml"]:
        config_result = caching_repo.load_config(f"{config_group}/{config_name}.{extension}")
        if config_result is None:
            continue
        try:
            return hydra_zen.get_target(config_result.config)  # type: ignore
        except TypeError:
            pass
    from hydra.plugins.config_source import ConfigLoadError

    try:
        config_node = _cs._load(f"{config_group}/{config_name}.yaml")
    except ConfigLoadError as error_yaml:
        try:
            config_node = _cs._load(f"{config_group}/{config_name}.yml")
        except ConfigLoadError:
            raise ConfigLoadError(
                f"Unable to find a config {config_group}/{config_name}.yaml or {config_group}/{config_name}.yml!"
            ) from error_yaml

    if "_target_" in config_node.node:
        target: str = config_node.node["_target_"]
        return hydra.utils.get_object(target)
        # module_name, _, class_name = target.rpartition(".")
        # module = importlib.import_module(module_name)
        # target = getattr(module, class_name)
        # return target

    # If it doesn't have a target, then assume that it's an inner dataclass like this:
    """
    ```python
    class Model:
        class HParams:
            ...
        def __init__(self, ...): # (with an arg of type HParams)
            ...
    """
    # NOTE: A bit hacky, might break.
    hparam_type = config_node.node._metadata.object_type
    target_type = get_outer_class(hparam_type)
    return target_type


def get_all_configs_in_group(group_name: str) -> list[str]:
    # note: here we're copying a bit of the internal code from Hydra so that we also get the
    # configs that are just yaml files, in addition to the configs we added programmatically to the
    # configstores.

    # names_yaml = cs.list(group_name)
    # names = [name.rpartition(".")[0] for name in names_yaml]
    # if "base" in names:
    #     names.remove("base")
    # return names

    return get_config_loader().get_group_options(group_name)


def get_all_configs_in_group_of_type(
    config_group: str,
    config_target_type: type | tuple[type, ...],
    include_subclasses: bool = True,
    excluding: type | tuple[type, ...] = (),
) -> list[str]:
    """Returns the names of all the configs in the given config group that have this target or a
    subclass of it."""
    config_names = get_all_configs_in_group(config_group)
    names_to_targets = {
        config_name: get_target_of_config(config_group, config_name)
        for config_name in config_names
    }

    names_to_types: dict[str, type] = {}
    for name, target in names_to_targets.items():
        if inspect.isclass(target):
            names_to_types[name] = target
            continue

        if (
            (inspect.isfunction(target) or inspect.ismethod(target))
            and (annotations := typing.get_type_hints(target))
            and (return_type := annotations.get("return"))
            and (inspect.isclass(return_type) or inspect.isclass(typing.get_origin(return_type)))
        ):
            # Resolve generic aliases if present.
            return_type = typing.get_origin(return_type) or return_type
            logger.debug(
                f"Assuming that the function {target} creates objects of type {return_type} based "
                f"on its return type annotation."
            )
            names_to_types[name] = return_type
            continue

        logger.warning(
            RuntimeWarning(
                f"Unable to tell what kind of object will be created by the target {target} of "
                f"config {name} in group {config_group}. This config will be skipped in tests."
            )
        )
    config_target_type = (
        config_target_type if isinstance(config_target_type, tuple) else (config_target_type,)
    )
    if excluding is not None:
        exclude = (excluding,) if isinstance(excluding, type) else excluding
        names_to_types = {
            name: object_type
            for name, object_type in names_to_types.items()
            if (
                not issubclass(object_type, exclude)
                if include_subclasses
                else object_type not in exclude
            )
        }

    def _matches_protocol(object: type, protocol: type) -> bool:
        return isinstance(object, protocol)  # todo: weird!

    compatible_config_names = []
    for name, object_type in names_to_types.items():
        if not include_subclasses:
            if object_type in config_target_type:
                compatible_config_names.append(name)
            continue
        for t in config_target_type:
            if (
                issubclass(t, typing.Protocol) and _matches_protocol(object_type, t)
            ) or issubclass(object_type, t):
                compatible_config_names.append(name)
                break

    return compatible_config_names


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
            pytest.mark.skip(
                reason="ResNets don't work with MNIST datasets because the image resolution is too small.",
                # raises=RuntimeError,
            )
        ]
        for resnet_config, mnist_dataset_config in itertools.product(
            get_all_configs_in_group_of_type("algorithm/network", torchvision.models.ResNet),
            get_all_configs_in_group_of_type(
                "datamodule", (MNISTDataModule, FashionMNISTDataModule)
            ),
        )
    },
}
"""Dict with some default marks to add to tests when some config combinations are present.

For example, ResNet networks can't be applied to the MNIST datasets.
"""


def run_for_all_configs_of_type(
    config_group: str, target_type: type, excluding: type | tuple[type, ...] = ()
):
    """Parametrizes a test to run with all the configs in the given group that have targets which
    are subclasses of the given type.

    For example:

    ```python
    @run_for_all_configs_of_type("algorithm", torch.nn.Module)
    def test_something_about_the_algorithm(algorithm: torch.nn.Module):
        ''' This test will run with all the configs in the 'algorithm' group that create nn.Modules! '''
    ```

    Concretely, this works by indirectly parametrizing the `f"{config_group}_config"` fixture.
    To learn more about indirect parametrization in PyTest, take a look at
    https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization
    """
    config_names = get_all_configs_in_group_of_type(
        config_group, target_type, include_subclasses=True, excluding=excluding
    )
    config_name_to_marks = {
        name: default_marks_for_config_name.get(name, []) for name in config_names
    }
    return run_for_all_configs_in_group(config_group, config_name_to_marks=config_name_to_marks)


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

    Parameters:
        arg_name_or_fixture: The name of the argument to parametrize, or a fixture to parametrize \
            indirectly.
        values: The values to be used to parametrize the test.

    Returns:
        A `pytest.MarkDecorator` that parametrizes the test with the given values only when the \
            argument is used (directly or indirectly) by the test.
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

    Parameters:
        group_name: List of datamodule names to use for tests. \
            By default, lists out the generic datamodules (the datamodules that aren't specific \
            to a single algorithm, for example the InfGendatamodules of WakeSleep.)

        config_name_to_marks: Dictionary from config names to pytest marks (e.g. \
            `pytest.mark.xfail`, `pytest.mark.skip`) to use for that particular config.
    """
    if config_name_to_marks is None:
        config_name_to_marks = {
            k: default_marks_for_config_name.get(k, [])
            for k in get_all_configs_in_group(group_name)
        }
    if "/" in group_name:
        group_name = group_name.replace("/", "_")
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


def total_vram_gb() -> float:
    """Returns the total VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return (
        sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        )
        / 1024**3
    )

"""Utility functions useful for testing."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib
import inspect
import os
import random
import typing
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Literal, TypeVar

import hydra_zen
import lightning
import numpy as np
import pytest
import torch
import yaml
from hydra.core.config_store import ConfigStore
from lightning import LightningModule
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.optim import Optimizer

from project.configs import Config
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.vision import VisionDataModule
from project.experiment import instantiate_trainer
from project.utils.env_vars import NETWORK_DIR
from project.utils.hydra_utils import get_attr, get_outer_class
from project.utils.types.protocols import (
    DataModule,
)
from project.utils.utils import get_device

logger = get_logger(__name__)

IN_GITHUB_CI = "GITHUB_ACTIONS" in os.environ
IN_SELF_HOSTED_GITHUB_CI = IN_GITHUB_CI and "self-hosted" in os.environ.get("RUNNER_LABELS", "")


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
}


def parametrized_fixture(name: str, values: Sequence, ids=None, **kwargs):
    """Small helper function that creates a parametrized pytest fixture for the given values.

    NOTE: When writing a fixture in a test class, use `ParametrizedFixture` instead.
    """

    @pytest.fixture(name=name, params=values, ids=ids or [f"{name}={v}" for v in values], **kwargs)
    def _parametrized_fixture(request: pytest.FixtureRequest):
        return request.param

    return _parametrized_fixture


class ParametrizedFixture[T]:
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


def get_config_loader():
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra._internal.utils import create_automatic_config_search_path

    search_path = create_automatic_config_search_path(
        calling_file=None, calling_module=None, config_path="pkg://project.configs"
    )
    config_loader = ConfigLoaderImpl(config_search_path=search_path)
    return config_loader


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


def get_all_algorithm_names() -> list[str]:
    """Retrieves the names of all the datamodules that are saved in the ConfigStore of Hydra."""
    return get_all_configs_in_group("algorithm")


def get_target_of_config(
    config_group: str, config_name: str, _cs: ConfigStore | None = None
) -> Callable:
    """Returns the class that is to be instantiated by the given config name.

    In the case of inner dataclasses (e.g. Model.HParams), this returns the outer class (Model).
    """
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
        module_name, _, class_name = target.rpartition(".")
        module = importlib.import_module(module_name)
        target = getattr(module, class_name)
        return target
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


def get_all_configs_in_group_with_target(group_name: str, some_type: type) -> list[str]:
    """Retrieves the names of all the configs in the given group that are used to construct objects
    of the given type."""
    config_names = get_all_configs_in_group(group_name)
    names_to_target = {
        config_name: get_target_of_config(group_name, config_name) for config_name in config_names
    }
    return [name for name, object_type in names_to_target.items() if object_type == some_type]


def get_all_configs_in_group_for_subclasses_of(group_name: str, base_class: type) -> list[str]:
    config_names = get_all_configs_in_group(group_name)
    names_to_targets = {
        config_name: get_target_of_config(group_name, config_name) for config_name in config_names
    }
    names_to_types: dict[str, type] = {}
    for name, target in names_to_targets.items():
        if inspect.isclass(target):
            names_to_types[name] = target
            continue

        if (
            inspect.isfunction(target)
            and (annotations := typing.get_type_hints(target))
            and (return_type := annotations.get("return"))
            and inspect.isclass(return_type)
        ):
            logger.info(
                f"Assuming that the function {target} creates objects of type {return_type} based "
                f"on its return type annotation."
            )
            names_to_types[name] = return_type
            continue

        logger.warning(
            RuntimeWarning(
                f"Unable to tell what kind of object will be created by the target {target} of config {name} in group {group_name}. This config will be skipepd in tests."
            )
        )

    return [
        name for name, object_type in names_to_types.items() if issubclass(object_type, base_class)
    ]


def get_vision_datamodule_names() -> list[str]:
    return get_all_configs_in_group_for_subclasses_of("datamodule", VisionDataModule)


def get_all_network_names() -> list[str]:
    """Retrieves the names of all the networks that are saved in the ConfigStore of Hydra.

    (This is the list of all the values that can be passed as the `network=<...>` argument on the
    command-line.)
    """
    return get_all_configs_in_group("network")


def get_network_names() -> list[str]:
    """Returns the name of all the networks."""
    network_names = get_all_network_names()
    return sorted(network_names)


def run_for_all_networks(
    network_names: list[str] | None = None,
    network_name_to_marks: dict[str, pytest.MarkDecorator | list[pytest.MarkDecorator]]
    | None = None,
    get_network_names_fn=get_network_names,
):
    """Apply this marker to a test to make it run with all available networks.

    The test should use the `network` fixture, either as an input argument to the test
    function or indirectly by using a fixture that depends on the network fixture.

    Parameters
    ----------
    network_names: list of network names to use.

    network_to_marks: Dictionary from network names to pytest marks (e.g. \
        `pytest.mark.xfail`, `pytest.mark.skip`) to use for that particular network.

    get_network_names_fn: Callable used to retrieve all the registered network configs. \
        By default, lists out the generic networks (the networks that aren't specific to a single \
        algorithm, for example the InfGenNetworks of WakeSleep.)

    Example
    -------

    ```python
    @run_with_all_networks({
        "some_network": pytest.mark.xfail(
            reason="this particular network's forward pass isn't reproducible atm."
        )
    })
    def test_network_output_is_reproducible(network: nn.Module, x: Tensor):
        # This test will be run with all networks, but is expected to fail when run with
        # the network whose name is 'some_network' (as in python main.py network=some_network).
        output_1 = network(x)
        output_2 = network(x)
        torch.testing.assert_close(output_1, output_2)
    ```
    """
    if network_names and network_name_to_marks:
        raise ValueError("Only one of `network_names` and `network_name_to_marks` can be set.")
    return run_for_all_configs_in_group(
        group_name="network",
        config_name_to_marks=network_name_to_marks
        or {name: [] for name in (network_names or get_network_names_fn())},
    )


def get_all_datamodule_names() -> list[str]:
    """Retrieves the names of all the datamodules that are saved in the ConfigStore of Hydra."""
    datamodules = get_all_configs_in_group("datamodule")
    # todo: automatically detect which ones are configs for ABCs and remove them?
    if "vision" in datamodules:
        datamodules.remove("vision")
    return datamodules


def get_all_datamodule_names_params():
    """Retrieves the names of all the datamodules that are saved in the ConfigStore of Hydra."""
    dm_names = get_all_datamodule_names()
    # NOTE: We put all the tests with the same datamodule in the same xdist group, so that when
    # doing distributed testing (with multiple processes on the same machine for now), tests with
    # the same datamodule are run in the same process. This is to save some memory and potential
    # redundant downloading/preprocessing.
    return [
        pytest.param(
            dm_name,
            marks=[
                pytest.mark.xdist_group(name=dm_name),
            ]
            + default_marks_for_config_name.get(dm_name, []),
        )
        for dm_name in dm_names
    ]


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
    if datamodule_names and datamodule_name_to_marks:
        raise ValueError(
            "Only one of `datamodule_names` and `datamodule_name_to_marks` can be set."
        )
    if datamodule_name_to_marks is None and datamodule_names:
        datamodule_name_to_marks = {
            datamodule_name: default_marks_for_config_name.get(datamodule_name, [])
            for datamodule_name in datamodule_names
        }
    return run_for_all_configs_in_group(
        group_name="datamodule",
        config_name_to_marks=datamodule_name_to_marks,
    )


def run_for_all_vision_datamodules():
    return run_for_all_subclasses_of("datamodule", VisionDataModule)


def run_for_all_subclasses_of(config_group: str, config_target_type: type):
    """Parametrizes a test to run with all the configs in the given group that have targets which
    are subclasses of the given type.

    For example:

    ```python
    @run_for_all_subclasses_of("network", torch.nn.Module)
    def test_something_about_the_network(network: torch.nn.Module):
        ''' This test will run with all the configs in the 'network' group that produce nn.Modules! '''
    ```
    """
    config_names = get_all_configs_in_group_for_subclasses_of(config_group, config_target_type)
    config_name_to_marks = {
        name: default_marks_for_config_name.get(name, []) for name in config_names
    }
    return run_for_all_configs_in_group(config_group, config_name_to_marks=config_name_to_marks)


def run_for_all_image_classification_datamodules():
    return run_for_all_subclasses_of("datamodule", ImageClassificationDataModule)


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
    if config_name_to_marks is None:
        config_name_to_marks = {
            k: default_marks_for_config_name.get(k, [])
            for k in get_all_configs_in_group(group_name)
        }
    # Parametrize the fixture (e.g. datamodule_name) indirectly, which will make it take each group
    # member (e.g. datamodule config name), each with a parameterized mark.
    return parametrize_when_used(
        f"{group_name}_name",
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


class AutoEncoder(LightningModule):
    def __init__(self, inf_network: nn.Module, gen_network: nn.Module):
        super().__init__()
        self.inf_network = inf_network
        self.gen_network = gen_network
        self.save_hyperparameters(ignore=("inf_network", "gen_network"))

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def reconstruct(self, input: Tensor) -> Tensor:
        return self.gen_network(self.inf_network(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.inf_network(input)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        return self.shared_step(batch, batch_index, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        return self.shared_step(batch, batch_index, phase="val")

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ) -> Tensor:
        x, _y = batch
        latents = self.inf_network(x)
        x_hat = self.gen_network(latents)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log(f"{phase}/loss", loss, on_epoch=True, prog_bar=True)
        return loss


class AutoEncoderClassifier(AutoEncoder):
    def __init__(
        self,
        inf_network: nn.Module,
        gen_network: nn.Module,
        num_classes: int,
        detach_latents: bool = False,
    ):
        super().__init__(inf_network, gen_network)
        self.num_classes = num_classes
        self.output_head = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_classes, device=get_device(inf_network))
        )
        self.detach_latents = detach_latents
        self.save_hyperparameters(ignore=("inf_network", "gen_network"))

    def forward(self, input: Tensor) -> Tensor:
        latents = self.inf_network(input)
        assert isinstance(latents, Tensor)
        output = self.output_head(latents.detach() if self.detach_latents else latents)
        assert isinstance(output, Tensor)
        return output

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ) -> Tensor:
        x, y = batch
        latents = self.inf_network(x)
        x_hat = self.gen_network(latents)
        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log(f"{phase}/recon_loss", recon_loss, prog_bar=True)

        logits = self.output_head(latents.detach() if self.detach_latents else latents)
        assert isinstance(logits, Tensor)
        ce_loss = torch.nn.functional.cross_entropy(logits, y)
        self.log(f"{phase}/ce_loss", ce_loss, prog_bar=True)
        accuracy = logits.argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/accuracy", accuracy, prog_bar=True)

        loss = recon_loss + ce_loss
        self.log(f"{phase}/loss", loss, prog_bar=True)
        return loss


class ImageClassifier(LightningModule):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.save_hyperparameters(ignore="network")

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, input: Tensor) -> Tensor:
        return self.network(input)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        return self.shared_step(batch, batch_index, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        return self.shared_step(batch, batch_index, phase="val")

    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ) -> Tensor:
        x, y = batch
        logits = self.network(x)
        assert isinstance(logits, Tensor)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log(f"{phase}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = logits.argmax(-1).eq(y).float().mean()
        self.log(f"{phase}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss


LightningModuleType = TypeVar("LightningModuleType", bound=LightningModule)


def train_module_for_tests(
    lightningmodule: LightningModuleType,
    base_experiment_config: Config,
    datamodule: DataModule,
) -> LightningModuleType:
    """Trains a lightningmodule to be used during tests.

    Avoids re-training by storing checkpoints in `temp`.
    """
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True, parents=False)

    assert isinstance(base_experiment_config.trainer, dict)
    trainer_config = copy.deepcopy(base_experiment_config.trainer)
    trainer_config["max_epochs"] = 2
    trainer_config["default_root_dir"] = None
    trainer_config.pop("logger", None)  # don't log to wandb during this "pre-training" phase.
    classifier_experiment_config = dataclasses.replace(
        base_experiment_config, trainer=trainer_config
    )
    trainer = instantiate_trainer(classifier_experiment_config)

    # NOTE: This is a roundabout way to get a dict with sorted keys from a DictConfig.
    config_dict = yaml.safe_load(OmegaConf.to_yaml(classifier_experiment_config, sort_keys=True))
    config_dict["algorithm"] = {
        "_target_": type(lightningmodule).__module__ + "." + type(lightningmodule).__qualname__,
        **lightningmodule.hparams,
    }
    config_hash = hashlib.md5(str(config_dict).encode(), usedforsecurity=False).hexdigest()[:16]

    experiment_config_hash = config_hash
    print(f"{experiment_config_hash=}")

    checkpoint_file = temp_dir / f"trained_classifier_{experiment_config_hash}.ckpt"

    if checkpoint_file.exists():
        logger.info(
            f"Reusing a previously-trained {type(lightningmodule).__name__} for the "
            "following tests."
        )
        trainer.fit(
            lightningmodule,
            datamodule=datamodule,  # type: ignore
            ckpt_path=str(checkpoint_file),
        )
        # trainer.save_checkpoint(checkpoint_file)
    else:
        logger.info("Training an image classifier to use during tests.")
        logger.debug(f"Cache miss for file {checkpoint_file}")
        trainer.fit(
            lightningmodule,
            datamodule=datamodule,  # type: ignore
        )
        logger.info("Finished training an image classifier for tests.")
        logger.info(f"Saving a checkpoint at {checkpoint_file} for future runs.")
        trainer.save_checkpoint(checkpoint_file)
    checkpoint_file.with_suffix(".yaml").write_text(yaml.dump(config_dict))

    return lightningmodule


def assert_same_shape_dtype_device(t1: Tensor, t2: Tensor):
    assert t1.shape == t2.shape, (t1.shape, t2.shape)
    assert t1.dtype == t2.dtype, (t1.dtype, t2.dtype)
    assert t1.device == t2.device, (t1.device, t2.device)


def assert_state_dicts_equal(module_a: nn.Module, module_b: nn.Module):
    state_dict_a = module_a.state_dict()
    state_dict_b = module_b.state_dict()

    initialized_params_a = {
        name: param
        for name, param in state_dict_a.items()
        if not isinstance(param, nn.UninitializedParameter)
    }
    initialized_params_b = {
        name: param
        for name, param in state_dict_b.items()
        if not isinstance(param, nn.UninitializedParameter)
    }

    torch.testing.assert_close(initialized_params_a, initialized_params_b)
    assert len(state_dict_a.keys()) == len(state_dict_b.keys())

    for (key_a, value_a), (key_b, value_b) in zip(state_dict_a.items(), state_dict_b.items()):
        assert key_a == key_b
        if isinstance(value_a, nn.UninitializedParameter):
            assert isinstance(value_b, nn.UninitializedParameter)
            continue
        parent_module = get_attr(module_a, ".".join(key_a.split(".")[:-1]))
        torch.testing.assert_close(
            value_a,
            value_b,
            msg=f"param {key_a!r} is different with the same seed! (parent: {parent_module})",
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
        # do a deepcopy just in case the libraries return the rng state "by reference" and keep
        # modifying it.
        return copy.deepcopy(cls())

    def set(self):
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


PARAM_WHEN_USED_MARK_NAME = "parametrize_when_used"

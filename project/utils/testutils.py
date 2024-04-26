"""Utility functions useful for testing."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, TypeVar

import hydra.errors
import pytest
import torch
import yaml
from hydra.core.config_store import ConfigStore
from lightning import LightningModule
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.optim import Optimizer

from project.configs.config import Config, cs
from project.configs.datamodule import DATA_DIR
from project.datamodules.bases.image_classification import ImageClassificationDataModule
from project.datamodules.bases.vision import VisionDataModule
from project.experiment import instantiate_trainer
from project.utils.hydra_utils import get_attr, get_outer_class
from project.utils.types import PhaseStr
from project.utils.types.protocols import DataModule
from project.utils.utils import get_device

SLOW_DATAMODULES = ["inaturalist", "imagenet32"]

default_marks_for_config_name: dict[str, list[pytest.MarkDecorator]] = {
    "imagenet32": [pytest.mark.slow],
    "inaturalist": [
        pytest.mark.slow,
        pytest.mark.xfail(
            not Path("/network/datasets/inat").exists(),
            strict=True,
            raises=hydra.errors.InstantiationException,
            reason="Expects to be run on the Mila cluster for now",
        ),
    ],
    "rl": [
        pytest.mark.xfail(
            strict=False,
            raises=AssertionError,
            # match="Shapes are not the same."
            reason="Isn't entirely deterministic yet.",
        ),
    ],
    "moving_mnist": [
        (pytest.mark.slow if not (DATA_DIR / "MovingMNIST").exists() else pytest.mark.timeout(5))
    ],
}
"""Dict with some default marks for some configs name."""


logger = get_logger(__name__)


def parametrized_fixture(name: str, values: Sequence, ids=None, **kwargs):
    """Small helper function that creates a parametrized pytest fixture for the given values.

    NOTE: When writing a fixture in a test class, use `ParametrizedFixture` instead.
    """

    @pytest.fixture(name=name, params=values, ids=ids or [f"{name}={v}" for v in values], **kwargs)
    def _parametrized_fixture(request: pytest.FixtureRequest):
        return request.param

    return _parametrized_fixture


class ParametrizedFixture:
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

    def __init__(self, values: list, name: str | None = None, **fixture_kwargs):
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


def get_all_configs_in_group(group_name: str) -> list[str]:
    names_yaml = cs.list(group_name)
    names = [name.rpartition(".")[0] for name in names_yaml]
    if "base" in names:
        names.remove("base")
    return names


def get_all_algorithm_names() -> list[str]:
    """Retrieves the names of all the datamodules that are saved in the ConfigStore of Hydra."""
    return get_all_configs_in_group("algorithm")


def get_type_for_config_name(config_group: str, config_name: str, _cs: ConfigStore = cs) -> type:
    """Returns the class that is to be instantiated by the given config name.

    In the case of inner dataclasses (e.g. Model.HParams), this returns the outer class (Model).
    """
    config_node = _cs._load(f"{config_group}/{config_name}.yaml")
    if "_target_" in config_node.node:
        target: str = config_node.node["_target_"]
        module_name, _, class_name = target.rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
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


def get_all_configs_in_group_for_subclasses_of(group_name: str, base_class: type) -> list[str]:
    datamodule_names = get_all_configs_in_group(group_name)
    names_to_types = {
        config_name: get_type_for_config_name(group_name, config_name)
        for config_name in datamodule_names
    }
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
    return get_all_configs_in_group("datamodule")


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
            + ([pytest.mark.slow] if dm_name in SLOW_DATAMODULES else []),
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


def run_for_all_subclasses_of(group_name: str, config_target_type: type):
    config_names = get_all_configs_in_group_for_subclasses_of(group_name, config_target_type)
    config_name_to_marks = {
        name: default_marks_for_config_name.get(name, []) for name in config_names
    }
    return run_for_all_configs_in_group(group_name, config_name_to_marks=config_name_to_marks)


def run_for_all_image_classification_datamodules():
    return run_for_all_subclasses_of("datamodule", ImageClassificationDataModule)


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

    return pytest.mark.parametrize(
        f"{group_name}_name",
        [
            pytest.param(
                config_name,
                marks=marks,
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

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, phase="val")

    def shared_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr) -> Tensor:
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

    def shared_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr) -> Tensor:
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

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, phase="val")

    def shared_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr) -> Tensor:
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


@contextmanager
def seeded(seed: int = 42):
    with torch.random.fork_rng():
        torch.random.manual_seed(seed)
        yield

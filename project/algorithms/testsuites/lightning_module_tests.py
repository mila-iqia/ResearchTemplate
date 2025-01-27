"""Suite of tests for an a `LightningModule`.

See the [project.algorithms.image_classifier_test][] module for an example of how to use this.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC
from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, overload

import lightning
import pytest
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from tensor_regression import TensorRegressionFixture

from project.configs.config import Config
from project.conftest import DEFAULT_SEED
from project.experiment import instantiate_trainer
from project.main import instantiate_algorithm, setup_logging
from project.utils.hydra_utils import resolve_dictconfig

logger = get_logger(__name__)

LightningModuleType = TypeVar("LightningModuleType", bound=LightningModule)


@pytest.mark.incremental  # https://docs.pytest.org/en/stable/example/simple.html#incremental-testing-test-steps
class LightningModuleTests(Generic[LightningModuleType], ABC):
    """Suite of generic tests for a LightningModule.

    Simply inherit from this class and decorate the class with the appropriate markers to get a set
    of decent unit tests that should apply to almost any LightningModule.

    See the [project.algorithms.image_classifier_test][] module for an example.

    Other ideas:
    - pytest-benchmark for regression tests on forward / backward pass / training step speed
    - pytest-profiling for profiling the training step? (pytorch variant?)
    - Dataset splits: check some basic stats about the train/val/test inputs, are they somewhat similar?
    - Define the input as a space, check that the dataset samples are in that space and not too
      many samples are statistically OOD?
    - Test to monitor distributed traffic out of this process?
        - Dummy two-process tests (on CPU) to check before scaling up experiments?
    """

    # algorithm_config: ParametrizedFixture[str]

    @pytest.fixture(scope="class")
    def experiment_config(
        self,
        experiment_dictconfig: DictConfig,
    ) -> Config:
        """The experiment configuration, with all interpolations resolved."""
        config = resolve_dictconfig(copy.deepcopy(experiment_dictconfig))
        return config

    @pytest.fixture(scope="class")
    def trainer(
        self,
        experiment_config: Config,
    ) -> lightning.Trainer:
        setup_logging(log_level=experiment_config.log_level)
        lightning.seed_everything(experiment_config.seed, workers=True)
        trainer = instantiate_trainer(experiment_config.trainer)
        assert isinstance(trainer, lightning.Trainer)
        return trainer

    @pytest.fixture(scope="class")
    def algorithm(
        self,
        experiment_config: Config,
        datamodule: lightning.LightningDataModule | None,
        trainer: lightning.Trainer,
        device: torch.device,
    ):
        """Fixture that creates the "algorithm" (a `LightningModule`)."""
        algorithm = instantiate_algorithm(experiment_config, datamodule=datamodule)
        assert isinstance(algorithm, LightningModule)
        with trainer.init_module(), device:
            # A bit hacky, but we have some tests that don't use a Trainer, and need the weights to
            # be initialized on the right device, but we don't have a Trainer yet.
            algorithm._device = device
            algorithm.configure_model()
        return algorithm

    @pytest.fixture(scope="class")
    def make_torch_deterministic(self):
        """Set torch to deterministic mode for unit tests that use the tensor_regression
        fixture."""
        mode_before = torch.get_deterministic_debug_mode()
        torch.set_deterministic_debug_mode("error")
        yield
        torch.set_deterministic_debug_mode(mode_before)

    @pytest.fixture(scope="class")
    def seed(self, request: pytest.FixtureRequest):
        """Fixture that seeds everything for reproducibility and yields the random seed used."""
        random_seed = getattr(request, "param", DEFAULT_SEED)
        assert isinstance(random_seed, int) or random_seed is None

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            lightning.seed_everything(random_seed, workers=True)
            yield random_seed

    @pytest.fixture(scope="class")
    def training_step_content(
        self,
        datamodule: lightning.LightningDataModule | None,
        algorithm: LightningModuleType,
        seed: int,
        accelerator: str,
        devices: int | list[int],
        tmp_path_factory: pytest.TempPathFactory,
    ):
        """Fixture that runs a training step and makes various things available for tests."""
        record_stuff_callback = GetStuffFromFirstTrainingStep()
        self.do_one_step_of_training(
            algorithm,
            datamodule,
            accelerator=accelerator,
            devices=devices,
            callbacks=[record_stuff_callback],
            tmp_path=tmp_path_factory.mktemp("training_step_content"),
        )
        assert record_stuff_callback.data is not None
        return record_stuff_callback.data

    def test_initialization_is_reproducible(
        self,
        training_step_content: StuffFromFirstTrainingStep,
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        """Check that the network initialization is reproducible given the same random seed."""
        tensor_regression.check(
            training_step_content.initial_state_dict,
            # todo: is this necessary? Shouldn't the weights be the same on CPU and GPU?
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=accelerator if accelerator not in ["auto", "gpu", "cuda"] else None,
            include_gpu_name_in_stats=False,
        )

    def test_forward_pass_is_reproducible(
        self,
        algorithm: LightningModuleType,
        training_step_content: StuffFromFirstTrainingStep,
        tensor_regression: TensorRegressionFixture,
    ):
        """Check that the forward pass is reproducible given the same input and random seed.

        Note: There could be more than one call to `forward` inside a training step. Here we only
        check the args/kwargs/outputs of the first `forward` call for now.
        """
        # Here we convert everything to dicts before saving to a file.
        # todo: make tensor-regression more flexible so it can handle tuples and lists in the dict.
        forward_pass_input = convert_list_and_tuples_to_dicts(training_step_content.forward_args)
        for forward_call_index, forward_kwargs in enumerate(training_step_content.forward_kwargs):
            for k, v in forward_kwargs.items():
                new_k = f"{forward_call_index}_{k}"
                if new_k in forward_pass_input:
                    # might not be necessary, but just in case.
                    new_k = f"{forward_call_index}_kwarg_{k}"
                assert new_k not in forward_pass_input
                forward_pass_input[new_k] = v

        forward_pass_output = convert_list_and_tuples_to_dicts(
            training_step_content.forward_outputs[0]
        )
        tensor_regression.check(
            {"input": forward_pass_input, "out": forward_pass_output},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance for changes.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            # todo: check if these values actually differ when run on cpu vs gpu.
            additional_label=next(algorithm.parameters()).device.type,
            include_gpu_name_in_stats=False,
        )

    def test_backward_pass_is_reproducible(
        self,
        training_step_content: StuffFromFirstTrainingStep,
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        """Check that the backward pass is reproducible given the same weights, inputs and random
        seed."""
        assert isinstance(training_step_content.grads, dict)
        assert isinstance(training_step_content.training_step_output, dict)
        # Here we convert everything to dicts before saving to a file.
        # todo: make tensor-regression more flexible so it can handle tuples and lists in the dict.
        batch = convert_list_and_tuples_to_dicts(training_step_content.batch)
        training_step_outputs = convert_list_and_tuples_to_dicts(
            training_step_content.training_step_output
        )
        tensor_regression.check(
            {
                "batch": batch,
                "grads": training_step_content.grads,
                "outputs": training_step_outputs,
            },
            # todo: this tolerance was mainly added for the jax example.
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance
            # todo: check if this actually differs between cpu / gpu.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=accelerator if accelerator not in ["auto", "gpu"] else None,
            include_gpu_name_in_stats=False,
        )

    def test_update_is_reproducible(
        self,
        algorithm: LightningModuleType,
        training_step_content: StuffFromFirstTrainingStep,
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        """Check that the weights after one step of training are the same given the same seed."""
        assert training_step_content.initial_state_dict
        tensor_regression.check(
            algorithm.state_dict(),
            # todo: is this necessary? Shouldn't the weights be the same on CPU and GPU?
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=accelerator if accelerator not in ["auto", "gpu", "cuda"] else None,
            include_gpu_name_in_stats=False,
        )

    def do_one_step_of_training(
        self,
        algorithm: LightningModuleType,
        datamodule: lightning.LightningDataModule | None,
        accelerator: str,
        devices: int | list[int] | Literal["auto"],
        callbacks: list[lightning.Callback],
        tmp_path: Path,
    ):
        """Performs one step of training.

        Overwrite this if you train your algorithm differently.
        """
        # NOTE: Here we create the trainer manually, but we could also
        # create it from the config (making sure to overwrite the right parameters to disable
        # checkpointing and logging to wandb etc.
        trainer = lightning.Trainer(
            accelerator=accelerator,
            callbacks=callbacks,
            devices=devices,
            fast_dev_run=True,
            enable_checkpointing=False,
            deterministic=True,
            default_root_dir=tmp_path,
        )
        trainer.fit(algorithm, datamodule=datamodule)
        return callbacks


@dataclasses.dataclass(frozen=True)
class StuffFromFirstTrainingStep:
    """Dataclass that holds information gathered from a training step and used in tests."""

    batch: Any | None = None
    """The input batch passed to the `training_step` method."""

    forward_args: list[tuple[Any, ...]] = dataclasses.field(default_factory=list)
    """The inputs args passed to each call to `forward` during the training step."""

    forward_kwargs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """The inputs kwargs apssed to each call to `forward` during the training step."""

    forward_outputs: list[Any] = dataclasses.field(default_factory=list)
    """The outputs of each call to the `forward` method during the training step."""

    initial_state_dict: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    """A copy of the state dict before the training step (moved to CPU)."""

    grads: dict[str, torch.Tensor | None] = dataclasses.field(default_factory=dict)
    """A copy of the gradients of the model parameters after the backward pass (moved to CPU)."""

    training_step_output: torch.Tensor | Mapping[str, Any] | None = None
    """The output of the `training_step` method."""


class GetStuffFromFirstTrainingStep(lightning.Callback):
    """Callback used in tests to get things from the first call to `training_step`."""

    def __init__(self):
        super().__init__()
        self.data: StuffFromFirstTrainingStep | None = None
        self._forward_hook_handle = None

    def on_train_batch_start(
        self, trainer: lightning.Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx != 0:
            return
        assert self.data is None
        self.data = StuffFromFirstTrainingStep(
            batch=batch,
            initial_state_dict={
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in pl_module.state_dict().items()
            },
        )
        assert self._forward_hook_handle is None
        self._forward_hook_handle = pl_module.register_forward_hook(
            self._save_forward_input_and_output,
            with_kwargs=True,
        )

    def _save_forward_input_and_output(
        self, module: LightningModule, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ):
        assert self.data is not None
        self.data.forward_args.append(args)
        self.data.forward_kwargs.append(kwargs)
        self.data.forward_outputs.append(output)

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx != 0:
            return
        assert self._forward_hook_handle is not None
        self._forward_hook_handle.remove()
        assert self.data is not None
        self.data = dataclasses.replace(self.data, training_step_output=outputs)

    def on_after_backward(self, trainer: lightning.Trainer, pl_module: LightningModule) -> None:
        if self.data is not None and self.data.grads:
            return  # already collected the gradients.

        assert self.data is not None
        for name, param in pl_module.named_parameters():
            self.data.grads[name] = (
                param.grad.detach().cpu().clone() if param.grad is not None else None
            )


@overload
def convert_list_and_tuples_to_dicts(
    value: torch.Tensor,
) -> torch.Tensor: ...


@overload
def convert_list_and_tuples_to_dicts(
    value: dict | tuple | list,
) -> dict[str, Any]: ...


def convert_list_and_tuples_to_dicts(
    value: torch.Tensor | dict | tuple | list,
) -> torch.Tensor | dict[str, Any]:
    """Converts all lists and tuples in a nested structure to dictionaries.

    >>> convert_list_and_tuples_to_dicts([1, 2, 3])
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts((1, 2, 3))
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts({"a": [1, 2, 3], "b": (4, 5, 6)})
    {'a': {'0': 1, '1': 2, '2': 3}, 'b': {'0': 4, '1': 5, '2': 6}}
    """

    def _inner(value):
        if isinstance(value, Mapping):
            return {k: _inner(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            # NOTE: Here we won't be able to distinguish between {"0": "bob"} and ["bob"]!
            # But that's not too bad.
            return {f"{i}": _inner(v) for i, v in enumerate(value)}
        return value

    if isinstance(value, Mapping):
        return _inner(value)
    if isinstance(value, list | tuple):
        return _inner(value)
    assert isinstance(value, dict | torch.Tensor), value
    return value

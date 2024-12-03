"""Suite of tests for an a `LightningModule`.

See the [project.algorithms.image_classifier_test][] module for an example of how to use this.
"""

from __future__ import annotations

import copy
from abc import ABC
from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import jax
import lightning
import pytest
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from tensor_regression import TensorRegressionFixture

from project.configs.config import Config
from project.conftest import DEFAULT_SEED
from project.experiment import instantiate_algorithm, instantiate_trainer, setup_logging
from project.trainers.jax_trainer import JaxTrainer
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.typing_utils import PyTree, is_sequence_of

logger = get_logger(__name__)

AlgorithmType = TypeVar("AlgorithmType", bound=LightningModule)


@pytest.mark.incremental  # https://docs.pytest.org/en/stable/example/simple.html#incremental-testing-test-steps
class LightningModuleTests(Generic[AlgorithmType], ABC):
    """Suite of generic tests for a LightningModule.

    Simply inherit from this class and decorate the class with the appropriate markers to get a set
    of decent unit tests that should apply to any LightningModule.

    See the [project.algorithms.image_classifier_test][] module for an example.

    Other ideas:
    - pytest-benchmark for regression tests on forward / backward pass / training step speed
    - pytest-profiling for profiling the training step? (pytorch variant?)
    - Dataset splits: check some basic stats about the train/val/test inputs, are they somewhat similar?
    - Define the input as a space, check that the dataset samples are in that space and not too
      many samples are statistically OOD?
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
    ) -> lightning.Trainer | JaxTrainer:
        setup_logging(log_level=experiment_config.log_level)
        lightning.seed_everything(experiment_config.seed, workers=True)
        return instantiate_trainer(experiment_config)

    @pytest.fixture(scope="class")
    def algorithm(
        self,
        experiment_config: Config,
        datamodule: lightning.LightningDataModule | None,
        trainer: lightning.Trainer | JaxTrainer,
        device: torch.device,
    ):
        """Fixture that creates the "algorithm" (a
        [LightningModule][lightning.pytorch.core.module.LightningModule])."""
        algorithm = instantiate_algorithm(experiment_config.algorithm, datamodule=datamodule)
        if isinstance(trainer, lightning.Trainer) and isinstance(
            algorithm, lightning.LightningModule
        ):
            with trainer.init_module(), device:
                # A bit hacky, but we have to do this because the lightningmodule isn't associated
                # with a Trainer.
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
        algorithm: AlgorithmType,
        seed: int,
        accelerator: str,
        devices: int | list[int],
        tmp_path_factory: pytest.TempPathFactory,
    ):
        """Check that the backward pass is reproducible given the same weights, inputs and random
        seed."""
        gradients_callback = GetStuffFromFirstTrainingStep()

        forward_pass_arg = []
        forward_pass_out = []

        def _save_forward_input_and_output(module: AlgorithmType, args, output):
            forward_pass_arg.append(args)
            forward_pass_out.append(output)

        with algorithm.register_forward_hook(_save_forward_input_and_output):
            self.do_one_step_of_training(
                algorithm,
                datamodule,
                accelerator=accelerator,
                devices=devices,
                callbacks=[gradients_callback],
                tmp_path=tmp_path_factory.mktemp("training_step_content"),
            )
        assert isinstance(gradients_callback.grads, dict)
        assert isinstance(gradients_callback.training_step_output, dict)
        return (algorithm, gradients_callback, forward_pass_arg, forward_pass_out)

    def test_initialization_is_reproducible(
        self,
        training_step_content: tuple[
            AlgorithmType, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        """Check that the network initialization is reproducible given the same random seed."""
        algorithm, *_ = training_step_content

        tensor_regression.check(
            algorithm.state_dict(),
            # todo: is this necessary? Shouldn't the weights be the same on CPU and GPU?
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=accelerator if accelerator not in ["auto", "gpu", "cuda"] else None,
            include_gpu_name_in_stats=False,
        )

    def test_forward_pass_is_reproducible(
        self,
        training_step_content: tuple[
            AlgorithmType, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
    ):
        """Check that the forward pass is reproducible given the same input and random seed."""
        algorithm, _test_callback, forward_pass_inputs, forward_pass_outputs = (
            training_step_content
        )
        # Here we convert everything to dicts before saving to a file.
        # todo: make tensor-regression more flexible so it can handle tuples and lists in the dict.
        forward_pass_input = convert_list_and_tuples_to_dicts(forward_pass_inputs[0])
        out = convert_list_and_tuples_to_dicts(forward_pass_outputs[0])
        tensor_regression.check(
            {"input": forward_pass_input, "out": out},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance for changes.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            # todo: check if these values actually differ when run on cpu vs gpu.
            additional_label=next(algorithm.parameters()).device.type,
            include_gpu_name_in_stats=False,
        )

    def test_backward_pass_is_reproducible(
        self,
        training_step_content: tuple[
            AlgorithmType, GetStuffFromFirstTrainingStep, list[Any], list[Any]
        ],
        tensor_regression: TensorRegressionFixture,
        accelerator: str,
    ):
        """Check that the backward pass is reproducible given the same weights, inputs and random
        seed."""
        _algorithm, test_callback, *_ = training_step_content
        assert isinstance(test_callback.grads, dict)
        assert isinstance(test_callback.training_step_output, dict)
        # Here we convert everything to dicts before saving to a file.
        # todo: make tensor-regression more flexible so it can handle tuples and lists in the dict.
        batch = convert_list_and_tuples_to_dicts(test_callback.batch)
        training_step_outputs = convert_list_and_tuples_to_dicts(
            test_callback.training_step_output
        )
        tensor_regression.check(
            {
                "batch": batch,
                "grads": test_callback.grads,
                "outputs": training_step_outputs,
            },
            # todo: this tolerance was mainly added for the jax example.
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance
            # todo: check if this actually differs between cpu / gpu.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=accelerator if accelerator not in ["auto", "gpu"] else None,
            include_gpu_name_in_stats=False,
        )

    @pytest.fixture(scope="session")
    def forward_pass_input(self, training_batch: PyTree[torch.Tensor], device: torch.device):
        """Extracts the model input from a batch of data coming from the dataloader.

        Overwrite this if your batches are not tuples of tensors (i.e. if your algorithm isn't a
        simple supervised learning algorithm like the example).
        """
        # By default, assume that the batch is a tuple of tensors.
        batch = training_batch

        def to_device(v):
            if hasattr(v, "to"):
                return v.to(device)
            return v

        batch = jax.tree.map(to_device, batch)

        if isinstance(batch, torch.Tensor | dict):
            return batch
        if not is_sequence_of(batch, torch.Tensor):
            raise NotImplementedError(
                "The basic test suite assumes that a batch is a tuple of tensors, as in the"
                f"supervised learning example, but the batch from the datamodule "
                f"is of type {type(batch)}. You need to override this method in your test class "
                "for the rest of the built-in tests to work correctly."
            )
        assert len(batch) >= 1
        input = batch[0]
        return input.to(device)

    def do_one_step_of_training(
        self,
        algorithm: AlgorithmType,
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


class GetStuffFromFirstTrainingStep(lightning.Callback):
    """Callback used in tests to get things from the first call to `training_step`."""

    def __init__(self):
        super().__init__()
        self.grads: dict[str, torch.Tensor | None] = {}
        self.batch: Any | None = None
        self.training_step_output: torch.Tensor | Mapping[str, Any] | None = None

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.batch is None:
            self.batch = batch
        if self.training_step_output is None:
            self.training_step_output = outputs

    def on_after_backward(self, trainer: lightning.Trainer, pl_module: LightningModule) -> None:
        super().on_after_backward(trainer, pl_module)
        if self.grads:
            return  # already collected the gradients.

        for name, param in pl_module.named_parameters():
            self.grads[name] = copy.deepcopy(param.grad)


def convert_list_and_tuples_to_dicts(value: Any) -> Any:
    """Converts all lists and tuples in a nested structure to dictionaries.

    >>> convert_list_and_tuples_to_dicts([1, 2, 3])
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts((1, 2, 3))
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts({"a": [1, 2, 3], "b": (4, 5, 6)})
    {'a': {'0': 1, '1': 2, '2': 3}, 'b': {'0': 4, '1': 5, '2': 6}}
    """
    if isinstance(value, Mapping):
        return {k: convert_list_and_tuples_to_dicts(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        # NOTE: Here we won't be able to distinguish between {"0": "bob"} and ["bob"]!
        # But that's not too bad.
        return {f"{i}": convert_list_and_tuples_to_dicts(v) for i, v in enumerate(value)}
    return value

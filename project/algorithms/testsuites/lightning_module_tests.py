"""Suite of tests for an a `LightningModule`.

See the [project.algorithms.image_classifier_test][] module for an example of how to use this.
"""

import copy
import inspect
from abc import ABC
from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, get_args

import jax
import lightning
import pytest
import torch
from lightning import LightningDataModule, LightningModule
from tensor_regression import TensorRegressionFixture

from project.configs.config import Config
from project.experiment import instantiate_algorithm
from project.utils.typing_utils import PyTree, is_sequence_of

logger = get_logger(__name__)

AlgorithmType = TypeVar("AlgorithmType", bound=LightningModule)


@pytest.mark.incremental
class LightningModuleTests(Generic[AlgorithmType], ABC):
    """Suite of generic tests for a LightningModule.

    Simply inherit from this class and decorate the class with the appropriate markers to get a set
    of decent unit tests that should apply to any LightningModule.

    See the [project.algorithms.image_classifier_test][] module for an example.
    """

    # algorithm_config: ParametrizedFixture[str]

    def forward_pass(self, algorithm: LightningModule, input: PyTree[torch.Tensor]):
        """Performs the forward pass with the lightningmodule, unpacking the inputs if necessary.

        Overwrite this if your algorithm's forward method is more complicated.
        """
        signature = inspect.signature(algorithm.forward)
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()):
            return algorithm(*input)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return algorithm(**input)
        return algorithm(input)

    def test_initialization_is_reproducible(
        self,
        experiment_config: Config,
        datamodule: lightning.LightningDataModule | None,
        seed: int,
        tensor_regression: TensorRegressionFixture,
        trainer: lightning.Trainer,
        device: torch.device,
    ):
        """Check that the network initialization is reproducible given the same random seed."""
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            algorithm = instantiate_algorithm(experiment_config.algorithm, datamodule=datamodule)
            assert isinstance(algorithm, lightning.LightningModule)
            # A bit hacky, but we have to do this because the lightningmodule isn't associated
            # with a Trainer here.
            with trainer.init_module(), device:
                algorithm._device = device
                algorithm.configure_model()

        tensor_regression.check(
            algorithm.state_dict(),
            # todo: is this necessary? Shouldn't the weights be the same on CPU and GPU?
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=next(algorithm.parameters()).device.type,
            include_gpu_name_in_stats=False,
        )

    def test_forward_pass_is_reproducible(
        self,
        forward_pass_input: Any,
        algorithm: AlgorithmType,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        """Check that the forward pass is reproducible given the same input and random seed."""
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            out = self.forward_pass(algorithm, forward_pass_input)
        # todo: make tensor-regression more flexible so it can handle tuples in the nested dict.
        forward_pass_input = convert_list_and_tuples_to_dicts(forward_pass_input)
        out = convert_list_and_tuples_to_dicts(out)
        tensor_regression.check(
            {"input": forward_pass_input, "out": out},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance for changes.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=next(algorithm.parameters()).device.type,
            include_gpu_name_in_stats=False,
        )

    def test_backward_pass_is_reproducible(
        self,
        datamodule: LightningDataModule,
        algorithm: AlgorithmType,
        seed: int,
        accelerator: str,
        devices: int | list[int],
        tensor_regression: TensorRegressionFixture,
        tmp_path: Path,
    ):
        """Check that the backward pass is reproducible given the same weights, inputs and random
        seed."""

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            gradients_callback = GetStuffFromFirstTrainingStep()
            self.do_one_step_of_training(
                algorithm,
                datamodule,
                accelerator=accelerator,
                devices=devices,
                callbacks=[gradients_callback],
                tmp_path=tmp_path,
            )
        # BUG: Fix issue in tensor_regression calling .numpy() on cuda tensors.
        assert isinstance(gradients_callback.grads, dict)
        assert isinstance(gradients_callback.outputs, dict)
        # todo: make tensor-regression more flexible so it can handle tuples and lists in the dict.
        batch = convert_list_and_tuples_to_dicts(gradients_callback.batch)
        outputs = convert_list_and_tuples_to_dicts(gradients_callback.outputs)
        tensor_regression.check(
            {
                "batch": batch,
                "grads": gradients_callback.grads,
                "outputs": outputs,
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
        datamodule: LightningDataModule,
        accelerator: str,
        devices: int | list[int] | Literal["auto"],
        callbacks: list[lightning.Callback],
        tmp_path: Path,
    ):
        """Performs one step of training.

        Overwrite this if you train your algorithm differently.
        """
        # TODO: Why are we creating the trainer here manually, why not load it from the config?
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


def _get_algorithm_class_from_generic_arg(
    cls: type[LightningModuleTests[AlgorithmType]],
) -> type[AlgorithmType]:
    """Retrieves the class under test from the class definition (without having to set a class
    attribute."""
    class_under_test = get_args(cls.__orig_bases__[0])[0]  # type: ignore
    if inspect.isclass(class_under_test) and issubclass(class_under_test, LightningModule):
        return class_under_test  # type: ignore

    # todo: Check if the class under test is a TypeVar, if so, check its bound.
    raise RuntimeError(
        "Your test class needs to pass the class under test to the generic base class.\n"
        "for example: `class TestMyAlgorithm(AlgorithmTests[MyAlgorithm]):`\n"
        f"(Got {class_under_test})"
    )


class GetStuffFromFirstTrainingStep(lightning.Callback):
    def __init__(self):
        super().__init__()
        self.grads: dict[str, torch.Tensor | None] = {}
        self.batch: Any | None = None
        self.outputs: torch.Tensor | Mapping[str, Any] | None = None

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
        if self.outputs is None:
            self.outputs = outputs

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

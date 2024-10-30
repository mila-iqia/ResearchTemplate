"""Suite of tests for an "algorithm".

See the [project.algorithms.example_test][] module for an example of how to use this.
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
from project.utils.typing_utils.protocols import DataModule

logger = get_logger(__name__)

# todo: potentially use an Algorithm protocol once the Example algo is type-checking OK against it.
AlgorithmType = TypeVar("AlgorithmType", bound=LightningModule)


def forward_pass(algorithm: LightningModule, input: PyTree[torch.Tensor]):
    """Performs the forward pass with the lightningmodule, unpacking the inputs if necessary."""
    if len(inspect.signature(algorithm.forward).parameters) == 1:
        return algorithm(input)
    assert isinstance(input, dict)
    return algorithm(**input)


@pytest.mark.incremental
class LearningAlgorithmTests(Generic[AlgorithmType], ABC):
    """Suite of unit tests for an "Algorithm" (LightningModule).

    Simply inherit from this class and decorate the class with the appropriate markers to get a set
    of decent unit tests that should apply to any LightningModule.

    See the [project.algorithms.example_test][] module for an example.
    """

    # algorithm_config: ParametrizedFixture[str]

    def test_initialization_is_deterministic(
        self,
        experiment_config: Config,
        datamodule: DataModule,
        seed: int,
    ):
        """Checks that the weights initialization is consistent given the a random seed."""

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            algorithm_1 = instantiate_algorithm(experiment_config.algorithm, datamodule)

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            algorithm_2 = instantiate_algorithm(experiment_config.algorithm, datamodule)

        torch.testing.assert_close(algorithm_1.state_dict(), algorithm_2.state_dict())

    def test_forward_pass_is_deterministic(
        self, forward_pass_input: Any, algorithm: AlgorithmType, seed: int
    ):
        """Checks that the forward pass output is consistent given the a random seed and a given
        input."""

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            out1 = forward_pass(algorithm, forward_pass_input)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            out2 = forward_pass(algorithm, forward_pass_input)

        torch.testing.assert_close(out1, out2)

    # @pytest.mark.timeout(10)
    def test_backward_pass_is_deterministic(
        self,
        datamodule: LightningDataModule,
        algorithm: AlgorithmType,
        seed: int,
        accelerator: str,
        devices: int | list[int] | Literal["auto"],
        tmp_path: Path,
    ):
        """Check that the backward pass is reproducible given the same input, weights, and random
        seed."""

        algorithm_1 = copy.deepcopy(algorithm)
        algorithm_2 = copy.deepcopy(algorithm)

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            gradients_callback = GetStuffFromFirstTrainingStep()
            self.do_one_step_of_training(
                algorithm_1,
                datamodule,
                accelerator,
                devices=devices,
                callbacks=[gradients_callback],
                tmp_path=tmp_path / "run1",
            )

        batch_1 = gradients_callback.batch
        gradients_1 = gradients_callback.grads
        training_step_outputs_1 = gradients_callback.outputs

        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            gradients_callback = GetStuffFromFirstTrainingStep()
            self.do_one_step_of_training(
                algorithm_2,
                datamodule,
                accelerator=accelerator,
                devices=devices,
                callbacks=[gradients_callback],
                tmp_path=tmp_path / "run2",
            )
        batch_2 = gradients_callback.batch
        gradients_2 = gradients_callback.grads
        training_step_outputs_2 = gradients_callback.outputs

        torch.testing.assert_close(batch_1, batch_2)
        torch.testing.assert_close(gradients_1, gradients_2)
        torch.testing.assert_close(training_step_outputs_1, training_step_outputs_2)

    def test_initialization_is_reproducible(
        self,
        experiment_config: Config,
        datamodule: DataModule,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        """Check that the network initialization is reproducible given the same random seed."""
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.random.manual_seed(seed)
            algorithm = instantiate_algorithm(experiment_config.algorithm, datamodule=datamodule)
        tensor_regression.check(
            algorithm.state_dict(),
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
            out = forward_pass(algorithm, forward_pass_input)

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
        batch = gradients_callback.batch
        if isinstance(batch, list | tuple):
            cpu_batch = {str(i): t.cpu() for i, t in enumerate(batch)}
        else:
            assert isinstance(batch, dict) and all(
                isinstance(v, torch.Tensor) for v in batch.values()
            )
            cpu_batch = {k: v.cpu() for k, v in batch.items()}
        tensor_regression.check(
            {
                # FIXME: This is ugly, and specific to the image classification example.
                "batch": cpu_batch,
                "grads": {
                    k: v.cpu() if v is not None else None
                    for k, v in gradients_callback.grads.items()
                },
                "outputs": {k: v.cpu() for k, v in gradients_callback.outputs.items()},
            },
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},  # some tolerance for the jax example.
            # Save the regression files on a different subfolder for each device (cpu / cuda)
            additional_label=next(algorithm.parameters()).device.type,
            include_gpu_name_in_stats=False,
        )

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # algorithm_under_test = _get_algorithm_class_from_generic_arg(cls)
        # # find all algorithm configs that create algorithms of this type.
        # configs_for_this_algorithm = get_all_configs_in_group_with_target(
        #     "algorithm", algorithm_under_test
        # )
        # # assert not hasattr(cls, "algorithm_config"), cls
        # cls.algorithm_config = ParametrizedFixture(
        #     name="algorithm_config",
        #     values=configs_for_this_algorithm,
        #     ids=configs_for_this_algorithm,
        #     ,
        # )

        # TODO: Could also add a parametrize_when_used mark to parametrize the datamodule, network,
        # etc, based on the type annotations of the algorithm constructor? For example, if an algo
        # shows that it accepts any LightningDataModule, then parametrize it with all the datamodules,
        # but if the algo says it only works with ImageNet, then parametrize with all the configs
        # that have the ImageNet datamodule as their target (or a subclass of ImageNetDataModule).

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
    cls: type[LearningAlgorithmTests[AlgorithmType]],
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

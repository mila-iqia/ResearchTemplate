import copy
import inspect
from abc import ABC
from logging import getLogger as get_logger
from typing import Generic, TypeVar, get_args

import lightning
import pytest
import torch
from lightning import LightningDataModule, LightningModule

from project.algorithms.example import ExampleAlgorithm
from project.configs.config import Config
from project.experiment import instantiate_algorithm
from project.utils.testutils import (
    ParametrizedFixture,
    fork_rng,
    get_all_configs_in_group_with_target,
    run_for_all_subclasses_of,
    seeded_rng,
)
from project.utils.types import PyTree, is_sequence_of
from project.utils.types.protocols import DataModule, Module

logger = get_logger(__name__)

AlgorithmType = TypeVar("AlgorithmType", bound=LightningModule)


@pytest.mark.incremental
class LearningAlgorithmTests(Generic[AlgorithmType], ABC):
    """Suite of unit tests for an "Algorithm" (LightningModule)."""

    algorithm_name: ParametrizedFixture[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        algorithm_under_test = _get_algorithm_class_from_generic_arg(cls)
        # find all algorithm configs that create algorithms of this type.
        configs_for_this_algorithm = get_all_configs_in_group_with_target(
            "algorithm", algorithm_under_test
        )
        cls.algorithm_name = ParametrizedFixture(
            name="algorithm_name",
            values=configs_for_this_algorithm,
            ids=str,
            scope="session",
        )

        # TODO: Could also add a parametrize_when_used mark to parametrize the datamodule, network,
        # etc, based on the type annotations of the algorithm constructor? For example, if an algo
        # shows that it accepts any LightningDataModule, then parametrize it with all the datamodules,
        # but if the algo says it only works with ImageNet, then parametrize with all the configs
        # that have the ImageNet datamodule as their target (or a subclass of ImageNetDataModule).

    def get_input_from_batch(self, batch: PyTree[torch.Tensor]):
        """Extracts the model input from a batch of data coming from the dataloader.

        Overwrite this if your batches are not tuples of tensors (i.e. if your algorithm isn't a
        simple supervised learning algorithm like the example).
        """
        # By default, assume that the batch is a tuple of tensors.
        if isinstance(batch, torch.Tensor):
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
        return input

    def test_initialization_is_deterministic(
        self,
        experiment_config: Config,
        datamodule: DataModule,
        network: torch.nn.Module,
        seed: int,
    ):
        with seeded_rng(seed):
            algorithm_1 = instantiate_algorithm(experiment_config, datamodule, network)

        with seeded_rng(seed):
            algorithm_2 = instantiate_algorithm(experiment_config, datamodule, network)

        torch.testing.assert_close(algorithm_1.state_dict(), algorithm_2.state_dict())

    def test_forward_pass_is_deterministic(
        self, training_batch: tuple[torch.Tensor, ...], network: Module, seed: int
    ):
        x = self.get_input_from_batch(training_batch)
        with fork_rng():
            out1 = network(x)
        with fork_rng():
            out2 = network(x)
        torch.testing.assert_close(out1, out2)

    @pytest.mark.timeout(10)
    def test_backward_pass_is_deterministic(
        self,
        datamodule: LightningDataModule,
        algorithm: LightningModule,
        seed: int,
        accelerator: str,
    ):
        class GetGradientsCallback(lightning.Callback):
            def __init__(self):
                super().__init__()
                self.grads: dict[str, torch.Tensor | None] = {}

            def on_after_backward(
                self, trainer: lightning.Trainer, pl_module: LightningModule
            ) -> None:
                super().on_after_backward(trainer, pl_module)
                if self.grads:
                    return  # already collected the gradients.

                for name, param in pl_module.named_parameters():
                    self.grads[name] = copy.deepcopy(param.grad)

        algorithm_1 = copy.deepcopy(algorithm)
        algorithm_2 = copy.deepcopy(algorithm)

        with seeded_rng(seed):
            gradients_callback = GetGradientsCallback()
            trainer = lightning.Trainer(
                accelerator=accelerator,
                callbacks=[gradients_callback],
                fast_dev_run=True,
                enable_checkpointing=False,
                deterministic=True,
            )
            trainer.fit(algorithm_1, datamodule=datamodule)
        gradients_1 = gradients_callback.grads

        with seeded_rng(seed):
            gradients_callback = GetGradientsCallback()
            trainer = lightning.Trainer(
                accelerator=accelerator,
                callbacks=[gradients_callback],
                fast_dev_run=True,
            )
            trainer.fit(algorithm_2, datamodule=datamodule)
        gradients_2 = gradients_callback.grads

        torch.testing.assert_close(gradients_1, gradients_2)


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


# @parametrize_when_used(network_name, ["fcnet", "resnet18"])
@run_for_all_subclasses_of("network", torch.nn.Module)
class TestExampleAlgo(LearningAlgorithmTests[ExampleAlgorithm]):
    """Tests for the `ExampleAlgorithm`."""

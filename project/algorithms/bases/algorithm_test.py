from __future__ import annotations

import contextlib
import copy
import inspect
import operator
import random
import sys
import typing
from collections.abc import Callable, Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
import pytest
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from typing_extensions import ParamSpec

from project.configs.config import Config, cs
from project.conftest import setup_hydra_for_tests_and_compose
from project.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from project.datamodules.vision.base import VisionDataModule
from project.experiment import (
    instantiate_datamodule,
    instantiate_network,
)
from project.main import main
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.maxpool_utils import has_maxunpool2d
from project.utils.tensor_regression import TensorRegressionFixture
from project.utils.testutils import (
    default_marks_for_config_name,
    get_all_datamodule_names_params,
    get_all_network_names,
    get_type_for_config_name,
)
from project.utils.types.protocols import DataModule

from .algorithm import Algorithm

logger = get_logger(__name__)
P = ParamSpec("P")

AlgorithmType = TypeVar("AlgorithmType", bound=Algorithm)


SKIP_OR_XFAIL = pytest.xfail if "-vvv" in sys.argv else pytest.skip
"""Either skips the test entirely (default) or tries to run it and expect it to fail (slower)."""

skip_test = pytest.mark.xfail if "-vvv" in sys.argv else pytest.mark.skip


class AlgorithmTests(Generic[AlgorithmType]):
    """Unit tests for an algorithm class.

    The algorithm creation is parametrized with all the datasets and all the networks, but the
    algorithm can choose to "opt out" of tests with certain datamodules / networks if they don't
    yet support them, using the `unsupported_datamodule_[names/types]` and
    `unsupported_network_[names/types]` attributes.

    NOTE: All the arguments to the test methods are created in fixtures below.

    In most cases, inheriting from this class and setting the class attributes should give you a
    set of tests that work somewhat well.

    You are obviously free to override any of the tests or fixtures for your own algorithm as you
    see fit.
    """

    algorithm_type: type[AlgorithmType]
    algorithm_name: ClassVar[str]

    unsupported_datamodule_names: ClassVar[list[str]] = []
    unsupported_datamodule_types: ClassVar[list[type[LightningDataModule]]] = []

    unsupported_network_names: ClassVar[list[str]] = []
    unsupported_network_types: ClassVar[list[type[nn.Module]]] = []

    # NOTE: These can also be used to specify explicitly which datamodules/networks this algorithm
    # is compatible with, however their use is discouraged, since we want to encourage researchers
    # to write algorithms that are as widely usable as possible.
    _supported_datamodule_names: ClassVar[list[str]] | None = None
    _supported_datamodule_types: ClassVar[list[type[LightningDataModule]]] | None = None
    _supported_network_names: ClassVar[list[str]] | None = None
    _supported_network_types: ClassVar[list[type[nn.Module]]] | None = None

    metric_name: ClassVar[str] = "train/loss"
    """The name of a scalar performance metric logged by the algorithm in its training_step."""

    lower_is_better: ClassVar[bool] = True
    """Whether decreasing the value of `metric_name` is an improvement or not."""

    # --------------------------------------- Tests -----------------------------------------------

    #

    @pytest.fixture(scope="class")
    def n_updates(self) -> int:
        """Returns the number of updates to perform before checking if the metric has improved.

        Some tests run a few updates on the same batch and then check that the performance metric
        (customizable via the `metric_name` attribute) improved.

        This fixture can be overwritten and customized to give a different value on a per-network
        or per-dataset basis, like so:

        ```python
        @pytest.fixture
        def n_updates(self, datamodule_name: str, network_name: str) -> int:
            if datamodule_name == "imagenet32":
                return 10
            return 3
        ```

        This allows overrides for specific datamodule/network combinations, for instance, some
        networks are not as powerful and require more updates to see an improvement in the metric.
        """
        # By default, perform 5 updates on a single batch before checking if the metric has
        # improved.
        return 5

    @pytest.mark.xfail(
        raises=NotImplementedError, reason="TODO: Implement this test.", strict=True
    )
    def test_loss_is_reproducible(
        self,
        algorithm: AlgorithmType,
        datamodule: LightningDataModule,
        seed: int,
        tensor_regression: TensorRegressionFixture,
    ):
        raise NotImplementedError(
            "TODO: Add tests that checks that the input batch, initialization and loss are "
            "reproducible."
        )

    def get_testing_callbacks(self) -> list[TestingCallback]:
        return [
            AllParamsShouldHaveGradients(),
        ]

    @pytest.mark.slow
    @pytest.mark.timeout(10)  # todo: make this much faster to run!
    def test_overfit_training_batch(
        self,
        algorithm: AlgorithmType,
        datamodule: LightningDataModule,
        accelerator: str,
        devices: list[int],
        n_updates: int,
        tmp_path: Path,
    ):
        testing_callbacks = self.get_testing_callbacks()
        if isinstance(datamodule, ImageClassificationDataModule):
            testing_callbacks.append(CheckBatchesAreTheSameAtEachStep(same_item_index=1))
        self._train(
            algorithm=algorithm,
            datamodule=datamodule,
            accelerator=accelerator,
            devices=devices,
            max_epochs=n_updates,
            overfit_batches=1,
            limit_val_batches=0.0,
            tmp_path=tmp_path,
            testing_callbacks=testing_callbacks,
        )

    def _train(
        self,
        algorithm: AlgorithmType,
        tmp_path: Path,
        testing_callbacks: Sequence[TestingCallback],
        # trainer arguments that are set from the fixtures.
        accelerator: str,
        devices: list[int],
        # default values that make sense during testing:
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
        # One of these must be passed:
        datamodule: LightningDataModule | None = None,
        train_dataloader: DataLoader | None = None,
        # Other arguments to be passed to the Trainer constructor:
        _trainer_type: Callable[P, Trainer] = Trainer,
        *trainer_args: P.args,
        **trainer_kwargs: P.kwargs,
    ):
        can_use_deterministic_mode = not has_maxunpool2d(algorithm) and not any(
            isinstance(mod, nn.modules.pooling._AdaptiveAvgPoolNd) for mod in algorithm.modules()
        )

        trainer = Trainer(
            *trainer_args,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            devices=devices,
            accelerator=accelerator,
            default_root_dir=tmp_path,
            callbacks=testing_callbacks.copy(),  # type: ignore
            # NOTE: Would be nice to be able to enforce this, but DTP uses nn.MaxUnpool2d.
            deterministic=True if can_use_deterministic_mode else "warn",
            **trainer_kwargs,
        )
        if datamodule is not None:
            trainer.fit(algorithm, datamodule=datamodule)
        else:
            trainer.fit(algorithm, train_dataloaders=train_dataloader)

        for callback in testing_callbacks:
            assert callback.was_executed

    @pytest.mark.xfail(
        reason="TODO: sort-of expected to fail because the tests for reproducibility of the loss "
        "(test_loss_is_reproducible) haven't been added yet."
    )
    @pytest.mark.slow  # todo: make this much faster to run!
    @pytest.mark.timeout(30)
    def test_experiment_reproducible_given_seed(
        self,
        datamodule_name: str,
        network_name: str,
        accelerator: str,
        devices: list[int] | int,
        tmp_path: Path,
        make_torch_deterministic: None,
        seed: int,
    ):
        """Tests that the experiment is reproducible given the same seed.

        NOTE: This test is close to using the command-line API, but not quite. If it were, we could
        launch jobs on the cluster to run the tests, which could be pretty neat!
        """

        if "resnet" in network_name and datamodule_name in ["mnist", "fashion_mnist"]:
            pytest.skip(reason="ResNet's can't be used on MNIST datasets.")

        algorithm_name = self.algorithm_name or self.algorithm_cls.__name__.lower()
        assert isinstance(algorithm_name, str)
        assert isinstance(datamodule_name, str)
        assert isinstance(network_name, str)
        all_overrides = [
            f"algorithm={algorithm_name}",
            f"network={network_name}",
            f"datamodule={datamodule_name}",
            "+trainer.limit_train_batches=3",
            "+trainer.limit_val_batches=3",
            "+trainer.limit_test_batches=3",
            "trainer.max_epochs=1",
            "seed=123",
            # NOTE: if we were to run the test in a slurm job, this wouldn't make sense.
            f"trainer.devices={devices}",
            f"trainer.accelerator={accelerator}",
        ]
        tmp_path_1 = tmp_path / "run_1"
        tmp_path_2 = tmp_path / "run_2"
        overrides_1 = all_overrides + [f"++trainer.default_root_dir={tmp_path_1}"]
        overrides_2 = all_overrides + [f"++trainer.default_root_dir={tmp_path_2}"]

        @contextlib.contextmanager
        def fork_rng():
            with torch.random.fork_rng():
                random_state = random.getstate()
                np_random_state = np.random.get_state()
                yield
                np.random.set_state(np_random_state)
                random.setstate(random_state)

        with (
            fork_rng(),
            setup_hydra_for_tests_and_compose(overrides_1, tmp_path=tmp_path_1) as config_1,
        ):
            performance_1 = main(config_1)

        with (
            fork_rng(),
            setup_hydra_for_tests_and_compose(overrides_2, tmp_path=tmp_path_2) as config_2,
        ):
            performance_2 = main(config_2)

        assert performance_1 == performance_2

    # TODOs:
    # - Finish reading https://www.pytorchlightning.ai/blog/effective-testing-for-machine-learning-systems
    # - Add more tests

    # ----------- Test Fixtures ----------- #

    @pytest.fixture(params=get_all_datamodule_names_params(), scope="class")
    def datamodule_name(self, request: pytest.FixtureRequest):
        """Fixture that gives the name of a datamodule to use."""
        datamodule_name = request.param

        if datamodule_name in default_marks_for_config_name:
            for marker in default_marks_for_config_name[datamodule_name]:
                request.applymarker(marker)

        self._skip_if_unsupported("datamodule", datamodule_name, skip_or_xfail=SKIP_OR_XFAIL)
        return datamodule_name

    @pytest.fixture(params=get_all_network_names(), scope="class")
    def network_name(self, request: pytest.FixtureRequest):
        """Fixture that gives the name of a network to use."""
        network_name = request.param

        if network_name in default_marks_for_config_name:
            for marker in default_marks_for_config_name[network_name]:
                request.applymarker(marker)

        self._skip_if_unsupported("network", network_name, skip_or_xfail=SKIP_OR_XFAIL)

        return network_name

    # TODO: This is a bit redundant with the `experiment_dictconfig` fixture from conftest which
    # does the same kind of thing. The only difference is that this one has access to the
    # attributes on the test class, so it's already parametrized and can know which
    # datamodules/networks are supported or not by this algorithm.

    @pytest.fixture(scope="class")
    def _hydra_config(
        self, datamodule_name: str, network_name: str, tmp_path_factory: pytest.TempPathFactory
    ) -> DictConfig:
        """Fixture that gives the Hydra configuration for an experiment that uses this algorithm,
        datamodule, and network.

        All overrides should have already been applied.
        """
        if "resnet" in network_name and datamodule_name in ["mnist", "fashion_mnist"]:
            pytest.skip(reason="ResNet's can't be used on MNIST datasets.")

        algorithm_name = self.algorithm_name
        with setup_hydra_for_tests_and_compose(
            all_overrides=[
                f"algorithm={algorithm_name}",
                f"datamodule={datamodule_name}",
                f"network={network_name}",
            ],
            tmp_path=tmp_path_factory.mktemp(
                f"testing_{algorithm_name}_{datamodule_name}_{network_name}"
            ),
        ) as config:
            return config

    # TODO: This very similar to the `experiment_config` fixture from conftest which does
    # the same kind of thing. The only difference is that this one has access to the supported /
    # unsupported datamodules and networks for this algorithm.
    @pytest.fixture(scope="class")
    def experiment_config(self, _hydra_config: DictConfig) -> Config:
        options = resolve_dictconfig(_hydra_config)
        assert isinstance(options, Config)
        return options

    @pytest.fixture(scope="class")
    def datamodule(self, experiment_config: Config) -> DataModule:
        """Creates the datamodule as it would be created with Hydra when using this algorithm."""
        datamodule = instantiate_datamodule(experiment_config)
        assert isinstance(datamodule, LightningDataModule)
        if self.unsupported_datamodule_types and isinstance(
            datamodule, tuple(self.unsupported_datamodule_types)
        ):
            SKIP_OR_XFAIL(
                reason=(
                    f"{self.algorithm_cls.__name__} doesn't support datamodules of "
                    f"type {type(datamodule)}"
                )
            )
        return datamodule

    @pytest.fixture(scope="class")
    def network(
        self, experiment_config: Config, datamodule: DataModule, device: torch.device
    ) -> nn.Module:
        network = instantiate_network(experiment_config, datamodule=datamodule)

        if self.unsupported_network_types and isinstance(
            network, tuple(self.unsupported_network_types)
        ):
            SKIP_OR_XFAIL(
                reason=(
                    f"{self.algorithm_cls.__name__} doesn't support networks of "
                    f"type {type(network)}"
                )
            )
        assert isinstance(network, nn.Module)
        return network.to(device=device)

    @pytest.fixture(scope="class")
    def hp(self, experiment_config: Config) -> Algorithm.HParams:  # type: ignore
        """The hyperparameters for the algorithm.

        NOTE: This should ideally be parametrized to test different hyperparameter settings.
        """
        return experiment_config.algorithm
        # return self.algorithm_cls.HParams()

    @pytest.fixture(scope="function")
    def algorithm_kwargs(
        self, datamodule: VisionDataModule, network: nn.Module, hp: Algorithm.HParams
    ):
        """Fixture that gives the keyword arguments to use to create the algorithm.

        NOTE: This should be further parametrized by base classes as needed.
        """
        return dict(datamodule=datamodule, network=copy.deepcopy(network), hp=hp)

    @pytest.fixture(scope="function")
    def algorithm(self, algorithm_kwargs: dict) -> AlgorithmType:
        return self.algorithm_cls(**algorithm_kwargs)

    @property
    def algorithm_cls(self) -> type[AlgorithmType]:
        """Returns the type of algorithm under test.

        If the `algorithm_type` attribute isn't set, then tries to detect the type of algo to test
        from the class definition. For example, `class TestMyAlgo(AlgorithmTests[MyAlgo]):` will
        return `MyAlgo` as the type of algorithm under test.
        """
        if not hasattr(self, "algorithm_type"):
            self.algorithm_type = self._algorithm_cls()
            return self.algorithm_type
        return self.algorithm_type

    @classmethod
    def _algorithm_cls(cls) -> type[AlgorithmType]:
        """Retrieves the class under test from the class definition (without having to set a class
        attribute."""
        import inspect
        from typing import get_args

        class_under_test = get_args(cls.__orig_bases__[0])[0]  # type: ignore
        if not (inspect.isclass(class_under_test) and issubclass(class_under_test, Algorithm)):
            raise RuntimeError(
                "Your test class needs to pass the class under test to the generic base class.\n"
                "for example: `class TestMyAlgorithm(AlgorithmTests[MyAlgorithm]):`\n"
                f"(Got {class_under_test})"
            )
        return class_under_test  # type: ignore

    def _skip_if_unsupported(
        self,
        group: Literal["network", "datamodule"],
        config_name: str,
        skip_or_xfail=SKIP_OR_XFAIL,
    ):
        unsupported_names: list[str] = getattr(self, f"unsupported_{group}_names")
        supported_names: list[str] | None = getattr(self, f"_supported_{group}_names")

        unsupported_types: list[type] = getattr(self, f"unsupported_{group}_types")
        supported_types: list[type] | None = getattr(self, f"_supported_{group}_types")

        if unsupported_names and supported_names:
            if not set(unsupported_names).isdisjoint(supported_names):
                raise RuntimeError(
                    f"The test class is setup incorrectly: it declares that the algorithm "
                    f"supports {group}={supported_names} but also that it doesn't support "
                    f"{group}={supported_names}. Please remove any overlap between these two "
                    f"fields."
                )

        if config_name in unsupported_names or (
            supported_names and config_name not in supported_names
        ):
            skip_or_xfail(
                reason=f"{self.algorithm_cls.__name__} doesn't support {group}={config_name}"
            )

        if not unsupported_types and not supported_types:
            return

        config_type: type = get_type_for_config_name(group, config_name, _cs=cs)
        if not inspect.isclass(config_type):
            config_return_type = typing.get_type_hints(config_type).get("return")
            if config_return_type and inspect.isclass(config_return_type):
                logger.warning(
                    f"Treating {config_type} as if it returns objects of type {config_return_type}"
                )
                config_type = config_return_type

        if unsupported_types and supported_types:
            if not set(unsupported_types).isdisjoint(supported_types):
                raise RuntimeError(
                    f"The test class is setup incorrectly: it declares that the algorithm "
                    f"supports {group}={supported_types} but also that it doesn't support "
                    f"{group}={supported_types}. Please remove any overlap between these two "
                    f"fields."
                )
        if issubclass(config_type, tuple(unsupported_types)):
            skip_or_xfail(
                reason=(
                    f"{self.algorithm_cls.__name__} doesn't support {group}={config_name} "
                    f"because {config_type} is a subclass of one of {unsupported_types}."
                )
            )
        if supported_types:
            assert all(inspect.isclass(t) for t in tuple(supported_types)), supported_types
        if supported_types:
            if inspect.isclass(config_type):
                if not issubclass(config_type, tuple(supported_types)):
                    skip_or_xfail(
                        reason=(
                            f"{self.algorithm_cls.__name__} doesn't support {group}={config_name} "
                            f"because {config_type} is not a subclass of one of {supported_types}."
                        )
                    )
            else:
                # config_type is not a class. Check if it is in the list of supported types anyway?
                # or check based on the return type maybe?
                logger.warning(
                    f"Unable to check if {config_type=} is within the list of supported types "
                    f"{supported_types=}!"
                )

        return True


class TestingCallback(Callback):
    """A Pytorch-Lightning Callback that checks something about the algorithm.

    It can collect stuff during any of the hooks, and should then check things in one of the
    hooks (for example in the `on_train_end` method).

    When the checks are done, it should call `self.done()` to indicate that the checks have been
    performed.
    """

    __test__ = False

    def __init__(self) -> None:
        super().__init__()
        self.was_executed: bool = False

    def done(self) -> None:
        self.was_executed = True


class GetMetricCallback(TestingCallback):
    """Simple callback used to store the value of a metric at each step of training."""

    def __init__(self, metric: str = "train/loss"):
        super().__init__()
        self.metric = metric
        self.metrics = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        assert self.metric in trainer.logged_metrics, (self.metric, trainer.logged_metrics.keys())
        metric_value = trainer.logged_metrics[self.metric]
        assert isinstance(metric_value, Tensor)
        self.metrics.append(metric_value.detach().item())


class MetricShouldImprove(GetMetricCallback):
    def __init__(
        self,
        metric: str = "train/loss",
        lower_is_better: bool | None = None,
        is_better_fn: Callable[[float, float], bool] | None = None,
    ):
        super().__init__(metric)
        if is_better_fn is None:
            is_better_fn = operator.lt if lower_is_better else operator.gt
        else:
            assert lower_is_better is None, (
                "If you pass a custom comparison function, you can't also pass `lower_is_better`",
            )
        self.comparison_fn = is_better_fn
        self.lower_is_better = lower_is_better

        self.num_training_steps = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self.num_training_steps = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.num_training_steps += 1

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert len(self.metrics) > 1
        m = self.metric
        # todo: could use something like the slope of Least-squares regression of that metric value
        # over time?
        assert self.comparison_fn(self.metrics[-1], self.metrics[0]), (
            f"metric {m}: didn't improve after {self.num_training_steps} steps:\n"
            f"before: {self.metrics[0]}, after: {self.metrics[-1]}",
        )
        self.done()


class GetGradientsCallback(TestingCallback):
    def __init__(self) -> None:
        super().__init__()
        self.was_executed = False
        self.gradients: dict[str, Tensor | None] = {}

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for name, parameter in pl_module.named_parameters():
            self.gradients[name] = (
                parameter.grad.clone().detach() if parameter.grad is not None else None
            )


class AllParamsShouldHaveGradients(GetGradientsCallback):
    def __init__(self, exceptions: Sequence[str] = ()) -> None:
        super().__init__()
        self.exceptions = exceptions

        self.gradients: dict[str, Tensor] = {}

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self.gradients.clear()

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger.debug(f"on_after_backward is being called at step {trainer.global_step}")
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is not None:
                self.gradients[name] = parameter.grad.clone().detach()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        parameters_with_nans = [
            name for name, param in pl_module.named_parameters() if param.isnan().any()
        ]
        assert not parameters_with_nans

        parameters_with_nans_in_grad = [
            name
            for name, param in pl_module.named_parameters()
            if param.grad is not None and param.grad.isnan().any()
        ]
        assert not parameters_with_nans_in_grad

        for name, parameter in pl_module.named_parameters():
            gradient = self.gradients.get(name)
            if not parameter.requires_grad:
                assert (
                    gradient is None
                ), f"Param {name} has a gradient when it doesn't require one!"
            elif any(name.startswith(exception) for exception in self.exceptions):
                pass
            else:
                assert (
                    gradient is not None
                ), f"param {name} doesn't have a gradient even though it requires one!"
                if (gradient == 0).all():
                    logger.warning(
                        RuntimeWarning(
                            f"Parameter {name} has a gradient of zero at step "
                            f"{trainer.global_step}!"
                        )
                    )
        self.done()


class CheckBatchesAreTheSameAtEachStep(TestingCallback):
    def __init__(self, same_item_index: int | str | None = None) -> None:
        """Checks that the batch (or a particular item) is exactly the same at each training step.

        Parameters
        ----------
        same_item_index: The index of the item in the batch that shouldn't change over time. \
            By default None, in which case the entire batch is expected to stay the same.
        """
        super().__init__()
        self.item_index = same_item_index
        self.previous_batch: Any = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.item_index is not None:
            batch = batch[self.item_index]
        if self.previous_batch is not None:
            torch.testing.assert_close(batch, self.previous_batch)
        self.previous_batch = batch

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # plt.figure()
        # plt.imsave("first.png", self.inputs[0][0].numpy())
        # plt.imsave("last.png", self.inputs[-1][0].numpy())
        self.done()
        # plt.imshow(self.inputs[0][0].numpy())
        # plt.imshow(self.inputs[-1][0].numpy())
        # plt.show()

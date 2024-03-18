from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from torch import Tensor, nn

from project.configs.config import Config, cs
from project.datamodules.datamodule import DataModule
from project.datamodules.vision import VisionDataModule
from project.utils.hydra_utils import get_instantiated_attr, get_outer_class
from project.main import main

from .algorithm import Algorithm

AlgorithmType = TypeVar("AlgorithmType", bound=Algorithm)


slow = pytest.mark.skipif(
    "-vvv" not in sys.argv, reason="Slow. Only runs when -vvv is passed."
)


def get_experiment_config(command_line_overrides: list[str]) -> Config:
    print(f"overrides: {' '.join(command_line_overrides)}")
    with initialize_config_module("project.configs"):
        config = compose(
            config_name="config",
            overrides=command_line_overrides,
        )

    config = OmegaConf.to_object(config)
    assert isinstance(config, Config)
    return config


def get_all_network_names() -> list[str]:
    """Retrieves the names of all the networks that are saved in the ConfigStore of Hydra.

    (This is the list of all the values that can be passed as the `network=<...>` argument on the
    command-line.)
    """
    network_names_yaml = cs.list("network")
    names = [name.rpartition(".")[0] for name in network_names_yaml]
    if "base" in names:
        names.remove("base")
    return names


def get_all_datamodule_names() -> list[str]:
    """Retrieves the names of all the datamodules that are saved in the ConfigStore of Hydra."""

    datamodule_names_yaml = cs.list("datamodule")
    names = [name.rpartition(".")[0] for name in datamodule_names_yaml]
    if "base" in names:
        names.remove("base")
    return names


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
            + ([slow] if dm_name in ["inaturalist", "imagenet32"] else []),
        )
        for dm_name in dm_names
    ]


class GetMetricCallback(Callback):
    """Simple callback used to store the value of a metric at each step of training."""

    def __init__(self, metric: str = "train/cross_entropy"):
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
        assert self.metric in trainer.logged_metrics, trainer.logged_metrics.keys()
        metric = trainer.logged_metrics[self.metric]
        assert isinstance(metric, Tensor)
        self.metrics += [metric.detach().item()]


class AlgorithmTests(Generic[AlgorithmType]):
    """Unit tests for an algorithm class.

    The algorithm creation is parametrized with all the datasets and all the networks, but the
    algorithm can choose to "opt out" of tests with certain datamodules / networks if they don't
    yet support them, using the `unsupported_datamodule_names` and `unsupported_network_names`
    attributes.
    """

    algorithm_type: type[AlgorithmType]
    algorithm_name: ClassVar[str]

    unsupported_datamodule_names: ClassVar[list[str]] = []
    unsupported_network_names: ClassVar[list[str]] = []

    metric_name: ClassVar[str] = "train/loss"
    """The main 'loss' metric to inspect to check if training is working."""

    # ----------- Test Fixtures ----------- #

    # TODO: Instead of parametrizing by everything and then skipping only some, only parametrize by
    # the supported ones. This isn't super important, but it would greatly reduce the number of
    # tests by only generating those that should be run.

    @pytest.fixture(params=get_all_datamodule_names_params(), scope="class")
    def datamodule_name(self, request: pytest.FixtureRequest):
        """Fixture that gives the name of a datamodule to use."""
        datamodule_name = request.param
        # NOTE: You can return a pytest.param(datamodule_name, marks=pytest.mark.skip) to skip a
        # particular datamodule, e.g. if it isn't supported by this algorithm.
        if datamodule_name in self.unsupported_datamodule_names:
            pytest.xfail(
                reason=f"{self.algorithm_cls.__name__} doesn't support the {datamodule_name} "
                f"datamodule"
            )
        return datamodule_name

    @pytest.fixture(params=get_all_network_names(), scope="class")
    def network_name(self, request: pytest.FixtureRequest):
        """Fixture that gives the name of a network to use."""
        network_name = request.param
        # NOTE: You can return a pytest.param(network_name, marks=pytest.mark.skip) to skip a
        # particular network, e.g. if it isn't supported by this algorithm.
        if network_name in self.unsupported_network_names:
            pytest.skip(
                reason=f"{self.algorithm_cls.__name__} doesn't support {network_name}"
            )
        return network_name

    @pytest.fixture(scope="class")
    def _hydra_config(
        self,
        datamodule_name: str,
        network_name: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> DictConfig:
        """Fixture that gives the Hydra configuration for an experiment that uses this algorithm,
        datamodule, and network.

        All overrides should have already been applied.
        """
        if "resnet" in network_name and datamodule_name in ["mnist", "fashion_mnist"]:
            pytest.skip(reason="ResNet's can't be used on MNIST datasets.")

        algorithm_name = self.algorithm_name
        with initialize_config_module(
            config_module="project.configs", version_base="1.2"
        ):
            config = compose(
                "config",
                overrides=[
                    f"algorithm={algorithm_name}",
                    f"datamodule={datamodule_name}",
                    f"network={network_name}",
                ],
                return_hydra_config=True,
            )

            # BUG: Weird errors with Hydra variable interpolation.. Setting these manually seems
            # to fix it for now..
            from hydra.conf import HydraHelpConf
            from hydra.core.hydra_config import HydraConfig

            with open_dict(config):
                # BUG: Getting some weird Hydra omegaconf error in unit tests:
                # "MissingMandatoryValue while resolving interpolation: Missing mandatory value:
                # hydra.job.num"
                config.hydra.job.num = 0
                config.hydra.hydra_help = HydraHelpConf(hydra_help="", template="")
                config.hydra.job.id = 0
                config.hydra.runtime.output_dir = str(
                    tmp_path_factory.mktemp(basename="output", numbered=True)
                )

            HydraConfig.instance().set_config(config)

            return config

    @pytest.fixture(scope="class")
    def hydra_options(self, _hydra_config: DictConfig) -> Config:
        # Important: Register this fancy little resolver here so we can get attributes of the
        # instantiated objects, not just the configs!
        instantiated_objects_cache: dict[str, Any] = {}
        OmegaConf.register_new_resolver(
            "instance_attr",
            functools.partial(
                get_instantiated_attr,
                _instantiated_objects_cache=instantiated_objects_cache,
            ),
        )
        options = OmegaConf.to_object(_hydra_config)
        assert isinstance(options, Config)
        return options

    @pytest.fixture(scope="class")
    def datamodule(self, hydra_options: Config) -> DataModule:
        """Creates the datamodule as it would be created with Hydra when using this algorithm."""
        datamodule = instantiate(hydra_options.datamodule)
        assert isinstance(datamodule, LightningDataModule)
        # assert isinstance(datamodule, DataModule)
        return datamodule

    @pytest.fixture
    def network(self, hydra_options: Config, datamodule: DataModule) -> nn.Module:
        network_hparams = hydra_options.network
        network_type: type[nn.Module] = get_outer_class(type(network_hparams))
        assert isinstance(
            network_hparams,
            network_type.HParams,  # type: ignore
        ), "HParams type should match net type"
        network = instantiate(hydra_options.network)
        # network = instantiate_network(network_hparams=network_hparams, datamodule=datamodule)
        # network = network_type(
        #     in_channels=datamodule.dims[0],
        #     n_classes=datamodule.num_classes,  # type: ignore
        #     hparams=network_hparams,
        # )
        assert isinstance(network, nn.Module)
        return network

    @pytest.fixture()
    def hp(self, hydra_options: Config) -> Algorithm.HParams:  # type: ignore
        """The hyperparameters for the algorithm.

        NOTE: This should ideally be parametrized to test different hyperparameter settings.
        """
        return hydra_options.algorithm
        # return self.algorithm_cls.HParams()

    @pytest.fixture
    def algorithm_kwargs(
        self, datamodule: VisionDataModule, network: nn.Module, hp: Algorithm.HParams
    ):
        """Fixture that gives the keyword arguments to use to create the algorithm.

        NOTE: This should be further parametrized by base classes as needed.
        """
        return dict(datamodule=datamodule, network=network, hp=hp)

    @pytest.fixture
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
        if not (
            inspect.isclass(class_under_test)
            and issubclass(class_under_test, Algorithm)
        ):
            raise RuntimeError(
                "Your test class needs to pass the class under test to the generic base class.\n"
                "for example: `class TestMyAlgorithm(AlgorithmTests[MyAlgorithm]):`\n"
                f"(Got {class_under_test})"
            )
        return class_under_test  # type: ignore

    @pytest.fixture(scope="class", params=["cpu", "gpu"])
    def accelerator(self, request: pytest.FixtureRequest):
        """Returns the accelerator to use during unit tests."""
        accelerator: str = request.param
        if accelerator == "gpu" and not torch.cuda.is_available():
            pytest.skip(reason="GPU not available")
        if accelerator == "cpu" and torch.cuda.is_available():
            if "-vvv" not in sys.argv:
                pytest.skip(
                    reason=(
                        "GPU is available and this would take a while on CPU."
                        "Only runs when -vvv is passed."
                    ),
                )
        return accelerator

    # ----------------- Tests -----------------

    def test_training_step_lowers_loss(
        self,
        algorithm: AlgorithmType,
        datamodule: LightningDataModule,
        accelerator: str,
    ):
        """Tests that the training loss on a batch of data decreases after a training step."""
        # TODO: Need to check that this still works for all algorithms.
        # x, y = training_batch
        # dataset = TensorDataset(x, y)
        # dataloader = DataLoader(dataset, batch_size=x.shape[0])

        my_callback = GetMetricCallback(self.metric_name)
        trainer = Trainer(
            # limit_train_batches=2,
            max_epochs=2,
            overfit_batches=1,
            log_every_n_steps=1,
            logger=False,
            enable_checkpointing=False,
            devices=1,  # TODO: Test with multiple GPUs.
            accelerator=accelerator,
            callbacks=[my_callback],
            # NOTE: Would be nice to be able to enforce this, but DTP uses nn.MaxUnpool2d.
            deterministic="warn",
        )
        assert algorithm.datamodule is datamodule
        assert isinstance(algorithm.datamodule, LightningDataModule)
        trainer.fit(algorithm, datamodule=algorithm.datamodule)
        assert len(my_callback.metrics) == 2
        assert my_callback.metrics[0] > my_callback.metrics[1]

    @slow()
    def test_experiment_reproducible_given_seed(
        self, datamodule_name: str, network_name: str, tmp_path: Path
    ):
        """Tests that the experiment is reproducible given the same seed."""
        algorithm_name = self.algorithm_name or self.algorithm_cls.__name__.lower()
        assert isinstance(algorithm_name, str)
        assert isinstance(datamodule_name, str)
        assert isinstance(network_name, str)
        all_overrides = [
            f"algorithm={algorithm_name}",
            f"network={network_name}",
            f"datamodule={datamodule_name}",
            # "debug=overfit",  # TODO: Update this. This won't work.
            "+trainer.limit_train_batches=3",
            "+trainer.limit_val_batches=3",
            "+trainer.limit_test_batches=3",
            f"++trainer.default_root_dir={tmp_path}",
            "trainer.max_epochs=1",
            "seed=123",
        ]
        print(f"overrides: {' '.join(all_overrides)}")
        with initialize_config_module("project.configs"):
            config = compose(
                config_name="config",
                overrides=all_overrides,
            )
            performance_1 = main(config)
            performance_2 = main(config)
            assert performance_1 == performance_2

    # TODOs:
    # - Finish reading https://www.pytorchlightning.ai/blog/effective-testing-for-machine-learning-systems
    # - Add more tests
    # - Re-add DTP as an algorithm.

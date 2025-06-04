"""Example showing how the test suite can be used to add tests for a new algorithm."""

import functools
import http
import http.server
import logging
import os
import pathlib
import socketserver
import sys
import webbrowser
from pathlib import Path

import lightning
import lightning.pytorch
import lightning.pytorch.loggers
import lightning.pytorch.profilers
import pytest
import torch
import torchvision
from pytest_benchmark.fixture import BenchmarkFixture

from project.algorithms.lightning_module_tests import LightningModuleTests
from project.configs import Config
from project.conftest import setup_with_overrides, skip_on_macOS_in_CI
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.main_test import experiment_commands_to_test
from project.utils.env_vars import DATA_DIR, SLURM_JOB_ID
from project.utils.testutils import IN_GITHUB_CI, run_for_all_configs_of_type

from .image_classifier import ImageClassifier

logger = logging.getLogger(__name__)

experiment_commands_to_test.extend(
    [
        "experiment=example trainer.fast_dev_run=True",
        pytest.param(
            f"experiment=cluster_sweep_example "
            f"trainer/logger=[] "  # disable logging.
            f"trainer.fast_dev_run=True "  # make each job quicker to run
            f"hydra.sweeper.worker.max_trials=1 "  # limit the number of jobs that get launched.
            f"resources=gpu "
            f"cluster={'current' if SLURM_JOB_ID else 'mila'} ",
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    IN_GITHUB_CI,
                    reason="Remote launcher tries to do a git push, doesn't work in github CI.",
                ),
            ],
        ),
        pytest.param(
            "experiment=local_sweep_example "
            "trainer/logger=[] "  # disable logging.
            "trainer.fast_dev_run=True "  # make each job quicker to run
            "hydra.sweeper.worker.max_trials=2 ",  # Run a small number of trials.
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "experiment=profiling "
            "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
            "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
            "trainer.fast_dev_run=True ",  # make each job quicker to run
            marks=pytest.mark.slow,
        ),
        (
            "experiment=profiling algorithm=no_op "
            "datamodule=cifar10 "  # Run a small dataset instead of ImageNet (would take ~6min to process on a compute node..)
            "trainer/logger=tensorboard "  # Use Tensorboard logger because DeviceStatsMonitor requires a logger being used.
            "trainer.fast_dev_run=True "  # make each job quicker to run
        ),
    ]
)


@setup_with_overrides("algorithm=image_classifier datamodule=cifar10")
def test_example_experiment_defaults(config: Config) -> None:
    """Test to check that the datamodule is required (even when just an algorithm is set?!)."""

    assert config.algorithm["_target_"] == (
        ImageClassifier.__module__ + "." + ImageClassifier.__qualname__
    )

    assert isinstance(config.datamodule, CIFAR10DataModule)


# When the `transformers` library is installed, for example when NLP-related examples are included,
# then we don't want this "run for all subclasses of nn.Module" to match these NLP models.
try:
    from transformers import PreTrainedModel

    excluding = PreTrainedModel
except ImportError:
    excluding = ()


@skip_on_macOS_in_CI
@run_for_all_configs_of_type("algorithm", ImageClassifier)
@run_for_all_configs_of_type("datamodule", ImageClassificationDataModule)
@run_for_all_configs_of_type("algorithm/network", torch.nn.Module, excluding=excluding)
class TestImageClassifier(LightningModuleTests[ImageClassifier]):
    """Tests for the `ImageClassifier`.

    This runs all the tests included in the base class, with the given parametrizations:

    - `algorithm_config` will take the value `"image_classifier"`
        - This is because there is an `image_classifier.yaml` config file in project/configs/algorithms
          whose `_target_` is the `ImageClassifier`.
    - `datamodule_config` will take these values: `['cifar10', 'fashion_mnist', 'imagenet', 'inaturalist', 'mnist']`
        - These are all the configs whose target is an `ImageClassificationDataModule`.
    - Similarly, `network_config` will be parametrized by the names of all configs which produce an nn.Module,
      except those that would create a `PreTrainedModel` from HuggingFace.
        - This is currently the easiest way for us to say "any network for image classification.

    Take a look at the `LightningModuleTests` class if you want to see the actual test code.
    """

    @pytest.mark.slow
    def test_benchmark_fit_speed(
        self,
        algorithm: ImageClassifier,
        datamodule: ImageClassificationDataModule,
        tmp_path_factory: pytest.TempPathFactory,
        benchmark: BenchmarkFixture,
    ):
        """Runs a few training steps a few times to compare wall-clock time between revisions.

        This uses [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) to
        run a measure the time it takes to run a few training steps.
        """
        # NOTE: Here we run this test will all the datamodules and networks that are parametrized
        # on the class. If you wanted to run this test outside of this repo or with a specific
        # datamodule or network, you could simply do this directly:
        # from torch.optim import Adam  # type: ignore
        # datamodule = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=64)
        # algo = ImageClassifier(
        #     datamodule=datamodule,
        #     network=torchvision.models.resnet18(weights=None, num_classes=datamodule.num_classes),
        #     optimizer=functools.partial(Adam, lr=1e-3, weight_decay=1e-4),
        # ).cuda()

        if datamodule is not None:
            # Do the data preparation ahead of time.
            datamodule.prepare_data()

        def run_some_training_steps() -> float:
            run_dir = tmp_path_factory.mktemp("benchmark_training_speed")
            trainer = lightning.Trainer(
                max_epochs=2,
                limit_train_batches=10,
                limit_val_batches=2,
                num_sanity_val_steps=0,
                log_every_n_steps=2,  # Benchmark with or without logging?
                logger=[
                    lightning.pytorch.loggers.TensorBoardLogger(run_dir),
                    lightning.pytorch.loggers.WandbLogger(save_dir=run_dir, mode="offline"),
                ],
                devices=1,
                accelerator="auto",
                default_root_dir=run_dir,
            )
            logger.info(f"Trainer log dir: {trainer.log_dir}")
            trainer.fit(algorithm, datamodule=algorithm.datamodule)
            train_metrics = trainer.logged_metrics
            assert isinstance(train_metrics, dict)
            train_acc = train_metrics["train/accuracy"]
            assert isinstance(train_acc, torch.Tensor)
            return train_acc.item()

        benchmark(run_some_training_steps)


def test_profile_training(tmp_path: Path) -> None:
    """Runs a little bit of training with the PyTorch Profiler (managed by Lightning).

    Outputs a trace file that can be viewed in the browser at `ui.perfetto.dev`.
    When run with `-vvv`, this will open the trace file in a new browser tab.
    """
    datamodule = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=64)
    from torch.optim import Adam  # type: ignore

    algo = ImageClassifier(
        datamodule=datamodule,
        network=torchvision.models.resnet18(weights=None, num_classes=datamodule.num_classes),
        optimizer=functools.partial(Adam, lr=1e-3, weight_decay=1e-4),
    ).cuda()
    datamodule.prepare_data()

    run_dir = tmp_path

    trainer = lightning.Trainer(
        max_epochs=2,
        limit_train_batches=50,
        limit_val_batches=0,
        limit_test_batches=0,
        logger=[
            # DO we want to include logging and callbacks?
            lightning.pytorch.loggers.TensorBoardLogger(run_dir),
            lightning.pytorch.loggers.WandbLogger(save_dir=run_dir, mode="offline"),
        ],
        devices=1,
        accelerator="auto",
        log_every_n_steps=10,  # to see if logging has an impact on performance.
        profiler=lightning.pytorch.profilers.PyTorchProfiler(
            run_dir, filename="profiler_output.txt", export_to_chrome=True
        ),
        default_root_dir=run_dir,
    )
    logger.info(f"Trainer log dir: {trainer.log_dir}")

    trainer.fit(algo, datamodule=algo.datamodule)
    # Uncomment to also profile validation:
    # _metrics = trainer.validate(algo, datamodule=algo.datamodule)

    logger.info(f"Trainer log dir: {trainer.log_dir}")
    assert trainer.log_dir is not None
    print(f"Trace files are in {run_dir}:")
    for file in run_dir.glob("*.trace.json"):
        print(f"  {file.name}")
        # If running tests in very verbose mode (-vvv), then open the trace file in a browser tab.
        if "-vvv" in sys.argv:
            url = _host_perfetto_trace_file(file)
            webbrowser.open(url, new=2)  # Open in a new tab, if possible.


def _host_perfetto_trace_file(path: os.PathLike | str):
    """Yanked out of the jax codebase.

    Very useful.
    """
    # ui.perfetto.dev looks for files hosted on `127.0.0.1:9001`. We set up a
    # TCP server that is hosting the `perfetto_trace.json.gz` file.
    port = 9001
    orig_directory = pathlib.Path.cwd()
    directory, filename = os.path.split(path)
    url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{filename}"
    try:
        os.chdir(directory)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", port), _PerfettoServer) as httpd:
            print(f"Open URL in browser: {url}")
            # Once ui.perfetto.dev acquires trace.json from this server we can close
            # it down.
            while httpd.__dict__.get("last_request") != "/" + filename:
                httpd.handle_request()
    finally:
        os.chdir(orig_directory)
    return url


class _PerfettoServer(http.server.SimpleHTTPRequestHandler):
    """Yanked from jax codebase: Handles requests from `ui.perfetto.dev` for the `trace.json`"""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        return super().end_headers()

    def do_GET(self):
        self.server.last_request = self.path  # type: ignore
        return super().do_GET()

    def do_POST(self):
        self.send_error(404, "File not found")

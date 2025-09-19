# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "hydra-zen",
#     "lightning",
#     "tensorboard",
#     "torch-tb-profiler",
#     "torchvision",
#     "tqdm",
#     "wandb",
# ]
# ///
import argparse
import functools
import http
import http.server
import logging
import os
import pathlib
import shutil
import socketserver
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path

import lightning
import lightning.pytorch
import lightning.pytorch.loggers
import lightning.pytorch.profilers
import optree
import pytest
import torch
import torchvision
import tqdm
import wandb
from torch.optim import Adam  # type: ignore
from torch.profiler import ProfilerAction, profile, schedule

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.algorithms.image_classifier import ImageClassifier
from project.datamodules.image_classification.cifar10 import CIFAR10DataModule
from project.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from project.main import REPO_ROOTDIR, setup_logging
from project.utils.env_vars import DATA_DIR

logger = logging.getLogger(__name__)


@pytest.fixture(
    scope="function",
    # params=[True, False],
    # ids=lambda v: "compiled" if v else "-",
)
def torch_compile(request: pytest.FixtureRequest) -> bool:
    """Fixture to parametrize the tests to run with or without torch.compile."""
    return getattr(request, "param", False)


@pytest.fixture(scope="function")
def algo_and_datamodule(torch_compile: bool):
    from torch.optim import Adam  # type: ignore

    datamodule = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=256)
    algo = ImageClassifier(
        datamodule=datamodule,
        network=torchvision.models.VisionTransformer(
            image_size=datamodule.dims[1],
            patch_size=4,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            num_classes=datamodule.num_classes,
        ),
        optimizer=functools.partial(Adam, lr=1e-3, weight_decay=1e-4),
    ).cuda()
    # Optional:
    if torch_compile:
        algo = torch.compile(algo)  # Compile the model to speed up training.
    return algo, datamodule


@pytest.fixture(scope="function")
def profiler_schedule() -> Callable[[int], torch.profiler.ProfilerAction]:
    """Returns a schedule for the PyTorch Profiler."""

    return schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Needs a GPU to run this test quickly.",
)
@pytest.mark.skipif(
    "-vvv" not in sys.argv,
    reason="This test is only useful when run with -vvv to open the trace file in a browser.",
)
def test_profile_lightning(
    algo_and_datamodule: tuple[ImageClassifier, ImageClassificationDataModule],
    profiler_schedule: Callable[[int], ProfilerAction],
    tmp_path: Path,
) -> None:
    """Runs a little bit of training with the PyTorch Profiler (managed by Lightning).

    Outputs a trace file that can be viewed in the browser at `ui.perfetto.dev`.
    When run with `-vvv`, this will open the trace file in a new browser tab.

    TODO: Alternatively, add the profiler to the Trainer used in `do_on_step_of_training` and reuse
    the output of the "training_step_content" fixture, to piggyback on the existing data instead of
    re-running some training steps.
    Other ideas:
    - Add a `training_loop_content` for tests that want stuff from a short training loop.
    """

    # torch.cuda.set_sync_debug_mode("warn")
    algo, datamodule = algo_and_datamodule
    lightning_training_loop(
        algo, datamodule, num_epochs=2, run_dir=tmp_path, profiler_schedule=profiler_schedule
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Needs a GPU to run this test quickly.",
)
@pytest.mark.skipif(
    "-vvv" not in sys.argv,
    reason="This test is only useful when run with -vvv to open the trace file in a browser.",
)
def test_profile_simple_train_loop(
    algo_and_datamodule: tuple[ImageClassifier, ImageClassificationDataModule],
    profiler_schedule: Callable[[int], ProfilerAction],
    tmp_path: Path,
):
    algo, datamodule = algo_and_datamodule
    simple_train_loop(
        algo, datamodule, num_epochs=2, run_dir=tmp_path, profiler_schedule=profiler_schedule
    )


def lightning_training_loop(
    algo: ImageClassifier,
    datamodule: ImageClassificationDataModule,
    num_epochs: int,
    profiler_schedule: Callable[[int], ProfilerAction],
    run_dir: Path,
):
    datamodule.prepare_data()
    run_dir = run_dir
    trainer = lightning.Trainer(
        max_epochs=num_epochs,
        limit_val_batches=0,
        limit_test_batches=0,
        callbacks=[
            # TODO: Investigate, this might be having a very bad impact on performance!
            # ClassificationMetricsCallback.attach_to(algo, num_classes=datamodule.num_classes),
            MeasureSamplesPerSecondCallback(),
            # Alternative to the Lightning wrapper around the pytorch profiler. Traces are simpler.
            # _ManualProfilerCallback(run_dir, schedule=profiler_schedule),
        ],
        logger=[
            # DO we want to include logging and callbacks?
            # lightning.pytorch.loggers.TensorBoardLogger(run_dir),
            lightning.pytorch.loggers.WandbLogger(save_dir=run_dir, mode="offline"),
        ],
        sync_batchnorm=False,
        # TODO: The multiprocessing of Lightning somehow leads to having two different `tmp_path`s
        # when running with `devices='auto'`! (and the rest of the test is run in each subprocess).
        # devices="auto",
        devices=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="auto",
        log_every_n_steps=1,  # to see if logging has an impact on performance.
        profiler=lightning.pytorch.profilers.PyTorchProfiler(
            run_dir,
            filename="profiler_output.txt",
            export_to_chrome=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir)),
            schedule=profiler_schedule,
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
        ),
        default_root_dir=run_dir,
    )
    logger.info(f"Trainer log dir: {trainer.log_dir}")

    trainer.fit(algo, datamodule=datamodule)
    wandb.finish()

    # Uncomment to also profile validation:
    # _metrics = trainer.validate(algo, datamodule=algo.datamodule)

    print(f"Trace files are in {run_dir}:")
    for file in run_dir.glob("*.trace.json"):
        print(f"  {file.name}")
        # If running tests in very verbose mode (-vvv), then open the trace file in a browser tab.
        # if "-vvv" in sys.argv:
        #     _host_perfetto_trace_file(file)


class _ManualProfilerCallback(lightning.pytorch.Callback):
    """The Pytorch Profiler option of Lightning makes traces that are very hard to read.

    This is simple.
    """

    def __init__(self, run_dir: Path, schedule: Callable[[int], ProfilerAction] | None = None):
        super().__init__()
        self.run_dir = run_dir
        if schedule is None:
            from torch.profiler import schedule as profiler_schedule

            schedule = profiler_schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2)
        self.profiler = profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir)),
            record_shapes=True,
            profile_memory=True,
        )

    def on_fit_start(self, *args, **kwargs) -> None:
        """Called when fit begins."""
        self.profiler.start()

    def on_train_batch_end(self, *args, **kwargs):
        """Called at the end of each training batch."""
        self.profiler.step()

    def on_fit_end(self, *args, **kwargs) -> None:
        """Called when fit ends."""
        self.profiler.__exit__(None, None, None)  # Stop the profiler and save the trace file.


def simple_train_loop(
    algo: ImageClassifier,
    datamodule: ImageClassificationDataModule,
    num_epochs: int,
    profiler_schedule: Callable[[int], ProfilerAction],
    run_dir: Path,
):
    optimizer = algo.configure_optimizers()
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    run = wandb.init(mode="offline")
    callbacks: list[lightning.Callback] = [
        # MeasureSamplesPerSecondCallback(),
        # TODO: Investigate, this might be having a very bad impact on performance!
        # ClassificationMetricsCallback.attach_to(algo, num_classes=datamodule.num_classes),
    ]
    with (
        profile(
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir)),
            # with_stack=False,  # makes things very verbose when run with pytest though.
            profile_memory=True,
            record_shapes=True,
        ) as p,
        run,
    ):
        # _accuracy = torch.zeros(1, device=algo.device, pin_memory=True)
        # _loss = torch.zeros(1, device=algo.device, pin_memory=True)

        for epoch in tqdm.tqdm(range(num_epochs), desc="Training", unit="Epochs", position=0):
            algo.train()
            for cb in callbacks:
                cb.on_train_epoch_start(None, algo)  # type: ignore

            for batch_index, batch in enumerate(
                tqdm.tqdm(  # type: ignore
                    train_dataloader,
                    desc=f"Train epoch {epoch}",
                    position=1,
                    leave=False,
                    unit_scale=train_dataloader.batch_size,
                    unit="Samples",
                )
            ):
                for cb in callbacks:
                    cb.on_train_batch_start(None, algo, batch, batch_index)  # type: ignore
                batch = optree.tree_map(lambda x: x.to(algo.device), batch)
                outputs = algo.training_step(batch, batch_index)
                loss = outputs["loss"]
                logits = outputs["logits"]
                y = outputs["y"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step_accuracy = logits.detach().argmax(-1).eq(y).float().mean()
                loss_mean = loss.detach().mean()
                # todo: do something fancy like this:
                # accuracy.copy_(step_accuracy, non_blocking=True)
                # loss.copy_(loss_mean, non_blocking=True)
                run.log(
                    {
                        "train/accuracy": step_accuracy.item(),
                        "train/loss": loss_mean.item(),
                    }
                )
                for cb in callbacks:
                    cb.on_train_batch_end(None, algo, None, batch, batch_index)
                p.step()

            for cb in callbacks:
                cb.on_train_epoch_end(None, algo)  # type: ignore


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
            webbrowser.open(url, new=2)  # Open in a new tab, if possible.

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


def main():
    parser = argparse.ArgumentParser("Benchmarking test for algorithms.")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Compile the model with torch.compile.",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--simple",
        dest="loop_type",
        const="simple",
        action="store_const",
        help="Run the simple training loop.",
    )
    g.add_argument(
        "--lightning",
        dest="loop_type",
        const="lightning",
        action="store_const",
        help="Run the Lightning training loop.",
    )
    args = parser.parse_args()
    loop: str = args.loop_type
    torch_compile: bool = args.compile

    num_epochs = 1
    datamodule = CIFAR10DataModule(data_dir=DATA_DIR, batch_size=64)
    algo = ImageClassifier(
        datamodule=datamodule,
        network=torchvision.models.VisionTransformer(
            image_size=datamodule.dims[1],
            patch_size=8,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            num_classes=datamodule.num_classes,
        ),
        optimizer=functools.partial(Adam, lr=1e-3, weight_decay=1e-4),
    ).cuda()
    # Optional:
    if torch_compile:
        algo = torch.compile(algo)  # Compile the model to speed up training.

    root_dir = REPO_ROOTDIR / "logs" / "benchmarking"

    profiler_schedule = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2)

    run_dir = root_dir / loop
    if run_dir.exists():
        logger.warning(f"Removing existing directory {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    if loop == "simple":
        simple_train_loop(
            algo,
            datamodule,
            num_epochs=num_epochs,
            profiler_schedule=profiler_schedule,
            run_dir=run_dir,
        )
    else:
        lightning_training_loop(
            algo,
            datamodule,
            num_epochs=num_epochs,
            profiler_schedule=profiler_schedule,
            run_dir=run_dir,
        )

    print("Trace files: ")
    for run_dir in [run_dir]:
        print(f"Run directory: {run_dir}")
        for file in run_dir.rglob("*.pt.trace.json"):
            print(f"  {file}")
            #
            # _host_perfetto_trace_file(file)


if __name__ == "__main__":
    setup_logging(log_level="INFO", global_log_level="WARNING")
    main()

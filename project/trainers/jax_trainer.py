from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chex
import flax.core
import flax.linen
import flax.struct
import jax
import jax.experimental
import jax.numpy as jnp
import lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
from hydra.core.hydra_config import HydraConfig
from typing_extensions import TypeVar

from project.configs.config import Config
from project.experiment import train
from project.utils.typing_utils.jax_typing_utils import jit

Ts = TypeVar("Ts", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode)
"""Type Variable for the training state."""

_B = TypeVar("_B", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode)
"""Type Variable for the batches produced (and consumed) by the algorithm."""

_MetricsT = TypeVar(
    "_MetricsT", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode, covariant=True
)
"""Type Variable for the metrics produced by the algorithm."""

__all__ = ["JaxModule", "JaxCallback", "JaxTrainer"]


@runtime_checkable
class JaxModule(Protocol[Ts, _B, _MetricsT]):
    """A protocol for algorithms that can be trained by the `JaxTrainer`.

    The `JaxRLExample` is an example that follows this structure and can be trained with a
    `JaxTrainer`.
    """

    def init_train_state(self, rng: chex.PRNGKey) -> Ts:
        """Create the initial training state."""
        raise NotImplementedError

    def get_batch(self, ts: Ts, batch_idx: int) -> tuple[Ts, _B]:
        """Produces a batch of data."""
        raise NotImplementedError

    def training_step(
        self, batch_idx: int, ts: Ts, batch: _B
    ) -> tuple[Ts, flax.struct.PyTreeNode]:
        """Update the training state using a "batch" of data."""
        raise NotImplementedError

    def eval_callback(self, ts: Ts) -> _MetricsT:
        """Perform evaluation and return metrics."""
        raise NotImplementedError


@train.register(JaxModule)
def train_jax_module(
    algorithm: JaxModule,
    /,
    *,
    trainer: JaxTrainer,
    config: Config,
    datamodule: None = None,
):
    if datamodule is not None:
        raise NotImplementedError(
            "The JaxTrainer doesn't yet support using a datamodule. For now, you should "
            f"return a batch of data from the {JaxModule.get_batch.__name__} method in your "
            f"algorithm."
        )

    if not isinstance(algorithm, JaxModule):
        raise TypeError(
            f"The selected algorithm ({algorithm}) doesn't implement the required methods of "
            f"a {JaxModule.__name__}, so it can't be used with the `{JaxTrainer.__name__}`. "
            f"Try to subclass {JaxModule.__name__} and implement the missing methods."
        )
    import jax

    rng = jax.random.key(config.seed)
    # TODO: Use ckpt_path argument to load the training state and resume the training run.
    assert config.ckpt_path is None
    ts, train_metrics = trainer.fit(algorithm, rng=rng)
    return algorithm, (ts, train_metrics)


class JaxCallback(flax.struct.PyTreeNode):
    def setup(self, trainer: JaxTrainer, module: JaxModule[Ts], stage: str, ts: Ts): ...
    def on_fit_start(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_fit_end(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_train_start(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_train_end(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_train_batch_start(
        self,
        trainer: JaxTrainer,
        pl_module: JaxModule[Ts, _B],
        batch: _B,
        batch_index: int,
        ts: Ts,
    ) -> None: ...
    def on_train_batch_end(
        self,
        trainer: JaxTrainer,
        module: JaxModule[Ts, _B],
        outputs: Any,
        batch: _B,
        batch_index: int,
        ts: Ts,
    ) -> None: ...
    def on_train_epoch_start(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_train_epoch_end(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_validation_epoch_start(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def on_validation_epoch_end(self, trainer: JaxTrainer, module: JaxModule[Ts], ts: Ts): ...
    def teardown(self, trainer: JaxTrainer, module: JaxModule[Ts], stage: str, ts: Ts): ...


class JaxTrainer(flax.struct.PyTreeNode):
    """A simplified version of the `lightning.Trainer` with a fully jitted training loop.

    ## Assumptions:

    - The algo object must match the `JaxModule` protocol (in other words, it should implement its
      methods).

    ## Training loop

    This is the training loop, which is fully jitted:

    ```python
    ts = algo.init_train_state(rng)

    setup("fit")
    on_fit_start()
    on_train_start()

    eval_metrics = []
    for epoch in range(self.max_epochs):
        on_train_epoch_start()

        for step in range(self.training_steps_per_epoch):

            batch = algo.get_batch(ts, step)

            on_train_batch_start()

            ts, metrics = algo.training_step(step, ts, batch)

            on_train_batch_end()

        on_train_epoch_end()

        # Evaluation "loop"
        on_validation_epoch_start()
        epoch_eval_metrics = self.eval_epoch(ts, epoch, algo)
        on_validation_epoch_start()

        eval_metrics.append(epoch_eval_metrics)

    return ts, eval_metrics
    ```

    ## Caveats

    - Some lightning callbacks can be used with this trainer and work well, but not all of them.
    - You can either use Regular pytorch-lightning callbacks, or use `jax.vmap` on the `fit` method,
      but not both.
      - If you want to use [jax.vmap][] on the `fit` method, just remove the callbacks on the
        Trainer for now.

    ## TODOs / ideas

    - Add a checkpoint callback with orbax-checkpoint?
    """

    max_epochs: int = flax.struct.field(pytree_node=False)

    training_steps_per_epoch: int = flax.struct.field(pytree_node=False)

    limit_val_batches: int = 0
    limit_test_batches: int = 0

    # TODO: Getting some errors with the schema generation for lightning.Callback and
    # lightning.pytorch.loggers.logger.Logger here if we keep the type annotation.
    callbacks: Sequence = dataclasses.field(metadata={"pytree_node": False}, default_factory=tuple)

    logger: Any | None = flax.struct.field(pytree_node=False, default=None)

    # accelerator: str = flax.struct.field(pytree_node=False, default="auto")
    # strategy: str = flax.struct.field(pytree_node=False, default="auto")
    # devices: int | str = flax.struct.field(pytree_node=False, default="auto")

    # path to output directory, created dynamically by hydra
    # path generation pattern is specified in `configs/hydra/default.yaml`
    # use it to store all files generated during the run, like checkpoints and metrics

    default_root_dir: str | Path | None = flax.struct.field(
        pytree_node=False,
        default_factory=lambda: HydraConfig.get().runtime.output_dir,
    )

    # State variables:
    # TODO: figure out how to cleanly store / update these.
    current_epoch: int = flax.struct.field(pytree_node=True, default=0)
    global_step: int = flax.struct.field(pytree_node=True, default=0)

    logged_metrics: dict = flax.struct.field(pytree_node=True, default_factory=dict)
    callback_metrics: dict = flax.struct.field(pytree_node=True, default_factory=dict)
    # todo: get the metrics from the callbacks?
    # lightning.pytorch.loggers.CSVLogger.log_metrics
    # TODO: Take a look at this method:
    # lightning.pytorch.callbacks.progress.rich_progress.RichProgressBar.get_metrics
    # return lightning.Trainer._logger_connector.progress_bar_metrics
    progress_bar_metrics: dict = flax.struct.field(pytree_node=True, default_factory=dict)

    verbose: bool = flax.struct.field(pytree_node=False, default=False)

    @functools.partial(jit, static_argnames=["skip_initial_evaluation"])
    def fit(
        self,
        algo: JaxModule[Ts, _B, _MetricsT],
        rng: chex.PRNGKey,
        train_state: Ts | None = None,
        skip_initial_evaluation: bool = False,
    ) -> tuple[Ts, _MetricsT]:
        """Full training loop in pure jax (a lot faster than when using pytorch-lightning).

        Unfolded version of `rejax.PPO.train`.

        Training loop in pure jax (a lot faster than when using pytorch-lightning).
        """

        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        train_state = train_state if train_state is not None else algo.init_train_state(rng)

        if self.progress_bar_callback is not None:
            if self.verbose:
                jax.debug.print("Enabling the progress bar callback.")
            jax.experimental.io_callback(self.progress_bar_callback.enable, ())

        self._callback_hook("setup", self, algo, ts=train_state, partial_kwargs=dict(stage="fit"))
        self._callback_hook("on_fit_start", self, algo, ts=train_state)
        self._callback_hook("on_train_start", self, algo, ts=train_state)

        if self.logger:
            jax.experimental.io_callback(
                lambda algo: self.logger and self.logger.log_hyperparams(hparams_to_dict(algo)),
                (),
                algo,
                ordered=True,
            )

        initial_evaluation: _MetricsT | None = None
        if not skip_initial_evaluation:
            initial_evaluation = algo.eval_callback(train_state)

        # Run the epoch loop `self.max_epoch` times.
        train_state, evaluations = jax.lax.scan(
            functools.partial(self.epoch_loop, algo=algo),
            init=train_state,
            xs=jnp.arange(self.max_epochs),  # type: ignore
            length=self.max_epochs,
        )

        if not skip_initial_evaluation:
            assert initial_evaluation is not None
            evaluations: _MetricsT = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluations,
            )

        if self.logger is not None:
            jax.block_until_ready((train_state, evaluations))
            # jax.debug.print("Saving...")
            jax.experimental.io_callback(
                functools.partial(self.logger.finalize, status="success"), ()
            )

        self._callback_hook("on_fit_end", self, algo, ts=train_state)
        self._callback_hook("on_train_end", self, algo, ts=train_state)
        self._callback_hook(
            "teardown", self, algo, ts=train_state, partial_kwargs={"stage": "fit"}
        )

        return train_state, evaluations

    # @jit
    def epoch_loop(self, ts: Ts, epoch: int, algo: JaxModule[Ts, _B, _MetricsT]):
        # todo: Some lightning callbacks try to get the "trainer.current_epoch".
        # FIXME: Hacky: Present a trainer with a different value of `self.current_epoch` to
        # the callbacks.
        # chex.assert_scalar_in(epoch, 0, self.max_epochs)
        # TODO: Can't just set current_epoch to `epoch` as `epoch` is a Traced value.
        # todo: need to have the callback take in the actual int value.
        # jax.debug.print("Starting epoch {epoch}", epoch=epoch)

        self = self.replace(current_epoch=epoch)  # doesn't quite work?
        ts = self.training_epoch(ts=ts, epoch=epoch, algo=algo)
        eval_metrics = self.eval_epoch(ts=ts, epoch=epoch, algo=algo)
        return ts, eval_metrics

    # @jit
    def training_epoch(self, ts: Ts, epoch: int, algo: JaxModule[Ts, _B, _MetricsT]):
        # Run a few training iterations
        self._callback_hook("on_train_epoch_start", self, algo, ts=ts)

        ts = jax.lax.fori_loop(
            0,
            self.training_steps_per_epoch,
            # drop training metrics for now.
            functools.partial(self.training_step, algo=algo),
            ts,
        )

        self._callback_hook("on_train_epoch_end", self, algo, ts=ts)
        return ts

    # @jit
    def eval_epoch(self, ts: Ts, epoch: int, algo: JaxModule[Ts, _B, _MetricsT]):
        self._callback_hook("on_validation_epoch_start", self, algo, ts=ts)

        # todo: split up into eval batch and eval step?
        eval_metrics = algo.eval_callback(ts=ts)

        self._callback_hook("on_validation_epoch_end", self, algo, ts=ts)

        return eval_metrics

    # @jit
    def training_step(self, batch_idx: int, ts: Ts, algo: JaxModule[Ts, _B, _MetricsT]):
        """Training step in pure jax (joined data collection + training).

        *MUCH* faster than using pytorch-lightning, but you lose the callbacks and such.
        """
        # todo: rename to `get_training_batch`?
        ts, batch = algo.get_batch(ts, batch_idx=batch_idx)

        self._callback_hook("on_train_batch_start", self, algo, batch, batch_idx, ts=ts)

        ts, metrics = algo.training_step(batch_idx=batch_idx, ts=ts, batch=batch)

        if self.logger is not None:
            # todo: Clean this up. logs metrics.
            jax.experimental.io_callback(
                lambda metrics, batch_index: self.logger
                and self.logger.log_metrics(
                    jax.tree.map(lambda v: v.mean(), metrics), batch_index
                ),
                (),
                dataclasses.asdict(metrics) if dataclasses.is_dataclass(metrics) else metrics,
                batch_idx,
            )

        self._callback_hook("on_train_batch_end", self, algo, metrics, batch, batch_idx, ts=ts)

        return ts

    ### Hooks to mimic those of lightning.Trainer

    def _callback_hook(
        self,
        hook_name: str,
        /,
        *hook_args,
        ts: Ts,
        partial_kwargs: dict | None = None,
        sharding: jax.sharding.SingleDeviceSharding | None = None,
        ordered: bool = True,
        **hook_kwargs,
    ):
        """Call a hook on all callbacks."""
        # with jax.disable_jit():
        for i, callback in enumerate(self.callbacks):
            # assert hasattr(callback, hook_name)

            method = getattr(callback, hook_name)
            if partial_kwargs:
                method = functools.partial(method, **partial_kwargs)
            if self.verbose:
                jax.debug.print(
                    "Epoch {current_epoch}/{max_epochs}: "
                    + f"Calling hook {hook_name} on callback {callback}"
                    + "{i}",
                    i=i,
                    current_epoch=self.current_epoch,
                    ordered=True,
                    max_epochs=self.max_epochs,
                )
            jax.experimental.io_callback(
                method,
                (),
                *hook_args,
                **({"ts": ts} if isinstance(callback, JaxCallback) else {}),
                **hook_kwargs,
                sharding=sharding,
                ordered=ordered if not isinstance(callback, JaxCallback) else False,
            )

    # Compat for RichProgressBar
    @property
    def is_global_zero(self) -> bool:
        return True

    @property
    def num_training_batches(self) -> int:
        return self.training_steps_per_epoch

    @property
    def loggers(self) -> list[lightning.pytorch.loggers.Logger]:
        if isinstance(self.logger, list | tuple):
            return list(self.logger)
        if self.logger is not None:
            return [self.logger]
        return []

    # @property
    # def progress_bar_metrics(self) -> dict[str, float]:

    #     return {}

    @property
    def progress_bar_callback(self) -> lightning.pytorch.callbacks.ProgressBar | None:
        for c in self.callbacks:
            if isinstance(c, lightning.pytorch.callbacks.ProgressBar):
                return c
        return None

    @property
    def state(self):
        from lightning.pytorch.trainer.states import (
            RunningStage,
            TrainerFn,
            TrainerState,
            TrainerStatus,
        )

        return TrainerState(
            fn=TrainerFn.FITTING,
            status=TrainerStatus.RUNNING,
            stage=RunningStage.TRAINING,
        )
        #     self._trainer.state.fn != "fit"
        #     or self._trainer.sanity_checking
        #     or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        # ):

    @property
    def sanity_checking(self) -> bool:
        from lightning.pytorch.trainer.states import RunningStage

        return self.state.stage == RunningStage.SANITY_CHECKING

    @property
    def training(self) -> bool:
        from lightning.pytorch.trainer.states import RunningStage

        return self.state.stage == RunningStage.TRAINING

    @property
    def log_dir(self) -> Path | None:
        # copied from lightning.Trainer
        if len(self.loggers) > 0:
            if not isinstance(
                self.loggers[0],
                lightning.pytorch.loggers.TensorBoardLogger | lightning.pytorch.loggers.CSVLogger,
            ):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir
        if dirpath:
            return Path(dirpath)
        return None


def hparams_to_dict(algo: flax.struct.PyTreeNode) -> dict:
    """Convert the learner struct to a serializable dict."""
    val = dataclasses.asdict(
        jax.tree.map(lambda arr: arr.tolist() if isinstance(arr, jnp.ndarray) else arr, algo)
    )
    val = jax.tree.map(lambda v: getattr(v, "__name__", str(v)) if callable(v) else v, val)
    return val

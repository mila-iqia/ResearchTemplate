from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Generic, ParamSpec, Protocol

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
from jax._src.sharding_impls import UNSPECIFIED, Device
from typing_extensions import TypeVar

P = ParamSpec("P")
Out = TypeVar("Out", covariant=True)


def jit(
    fn: Callable[P, Out],
    in_shardings=UNSPECIFIED,
    out_shardings=UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> Callable[P, Out]:
    """Small type hint fix for jax's `jit` (preserves the signature of the callable)."""
    return jax.jit(
        fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
    )


Ts = TypeVar("Ts", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode)
_B = TypeVar("_B", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode)
_MetricsT = TypeVar(
    "_MetricsT", bound=flax.struct.PyTreeNode, default=flax.struct.PyTreeNode, covariant=True
)


class JaxModule(Protocol[Ts, _B, _MetricsT]):
    def get_batch(self, ts: Ts, batch_idx: int) -> tuple[Ts, _B]: ...
    def training_step(
        self, batch_idx: int, ts: Ts, batch: _B
    ) -> tuple[Ts, flax.struct.PyTreeNode]: ...
    def eval_callback(self, ts: Ts) -> _MetricsT: ...
    def init_train_state(self, rng: chex.PRNGKey) -> Ts: ...


class JaxCallback(Generic[Ts, _B], flax.struct.PyTreeNode):
    def setup(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], stage: str, ts: Ts): ...
    def on_fit_start(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def on_fit_end(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def on_train_start(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def on_train_end(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
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
    def on_train_epoch_start(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def on_train_epoch_end(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def on_validation_epoch_start(
        self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts
    ): ...
    def on_validation_epoch_end(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], ts: Ts): ...
    def teardown(self, trainer: JaxTrainer, module: JaxModule[Ts, _B], stage: str, ts: Ts): ...


class JaxTrainer(flax.struct.PyTreeNode, Generic[Ts, _B, _MetricsT]):
    """Somewhat similar to a `lightning.Trainer`."""

    # num_epochs = np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int)
    max_epochs: int = flax.struct.field(pytree_node=False)

    # iteration_steps = algo.hp.num_envs * algo.hp.num_steps
    # num_iterations = np.ceil(algo.hp.eval_freq / iteration_steps).astype(int)
    training_steps_per_epoch: int

    # training_step_fn: Callable[[TrainState, int], tuple[TrainState, Any]]
    callbacks: Sequence[lightning.Callback | JaxCallback] = flax.struct.field(
        pytree_node=False, default_factory=list
    )

    logger: lightning.pytorch.loggers.Logger | None = flax.struct.field(
        pytree_node=False, default=None
    )

    # accelerator: str = flax.struct.field(pytree_node=False, default="auto")
    # strategy: str = flax.struct.field(pytree_node=False, default="auto")
    # devices: int | str = flax.struct.field(pytree_node=False, default="auto")

    # min_epochs: int

    # path to output directory, created dynamically by hydra
    # path generation pattern is specified in `configs/hydra/default.yaml`
    # use it to store all files generated during the run, like checkpoints and metrics
    default_root_dir: str | Path | None = flax.struct.field(
        pytree_node=False, default=""
    )  # ${hydra:runtime.output_dir}

    # State variables:
    # TODO: Figure out how to efficiently present these even when jit is turned off (currently
    # replacing self entirely).
    current_epoch: int = flax.struct.field(pytree_node=True, default=0)
    global_step: int = flax.struct.field(pytree_node=True, default=0)

    # TODO: Add a checkpoint callback with orbax-checkpoint?

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
            jax.experimental.io_callback(self.progress_bar_callback.enable, ())

        self._callback_hook("setup", self, algo, ts=train_state, partial_kwargs=dict(stage="fit"))
        self._callback_hook("on_fit_start", self, algo, ts=train_state)
        self._callback_hook("on_train_start", self, algo, ts=train_state)

        if self.logger:
            jax.experimental.io_callback(
                lambda algo: self.logger and self.logger.log_hyperparams(hparams_to_dict(algo)),
                (),
                algo,
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

    @jit
    # @functools.partial(jit, static_argnames=["epoch"])
    def epoch_loop(self, ts: Ts, epoch: int, algo: JaxModule[Ts, _B, _MetricsT]):
        # todo: Some lightning callbacks try to get the "trainer.current_epoch".
        # FIXME: Hacky: Present a trainer with a different value of `self.current_epoch` to
        # the callbacks.
        # chex.assert_scalar_in(epoch, 0, self.max_epochs)
        # TODO: Can't just set current_epoch to `epoch` as `epoch` is a Traced value.
        # todo: need to have the callback take in the actual int value.
        self = self.replace(current_epoch=epoch)  # doesn't quite work!
        ts = self.training_epoch(ts=ts, epoch=epoch, algo=algo)
        eval_metrics = self.eval_epoch(ts=ts, epoch=epoch, algo=algo)
        return ts, eval_metrics

    @jit
    # @functools.partial(jit, static_argnames=["epoch"])
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

    @jit
    # @functools.partial(jit, static_argnames="epoch")
    def eval_epoch(self, ts: Ts, epoch: int, algo: JaxModule[Ts, _B, _MetricsT]):
        self._callback_hook("on_validation_epoch_start", self, algo, ts=ts)

        # todo: split up into eval batch and eval step?
        eval_metrics = algo.eval_callback(ts=ts)

        self._callback_hook("on_validation_epoch_end", self, algo, ts=ts)

        return eval_metrics

    @jit
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
                metrics,
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
        ordered: bool = False,
        **hook_kwargs,
    ):
        """Call a hook on all callbacks."""
        for i, callback in enumerate(self.callbacks):
            assert hasattr(callback, hook_name)
            method = getattr(callback, hook_name)
            if partial_kwargs:
                method = functools.partial(method, **partial_kwargs)

            jax.experimental.io_callback(
                method,
                (),
                *hook_args,
                **({"ts": ts} if isinstance(callback, JaxCallback) else {}),
                **hook_kwargs,
                sharding=sharding,
                ordered=ordered,
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

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        # todo: get the metrics from the callbacks?
        # lightning.pytorch.loggers.CSVLogger.log_metrics
        # TODO: Take a look at this method:
        # lightning.pytorch.callbacks.progress.rich_progress.RichProgressBar.get_metrics
        # return lightning.Trainer._logger_connector.progress_bar_metrics
        return {}

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

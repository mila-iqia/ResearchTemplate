from __future__ import annotations

import dataclasses
import functools
import operator
import time
from collections.abc import Callable, Iterable
from logging import getLogger
from pathlib import Path
from typing import Any

import chex
import gymnax
import jax
import jax.numpy as jnp
import lightning
import numpy as np
import pytest
import rejax
import scipy.stats
import torch
import torch_jax_interop
from gymnax.environments.environment import Environment
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from tensor_regression import TensorRegressionFixture
from torch.utils.data import DataLoader
from typing_extensions import override

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.trainers.jax_trainer import JaxTrainer, hparams_to_dict
from project.utils.testutils import run_for_all_configs_of_type

from .jax_rl_example import (
    EvalMetrics,
    JaxRLExample,
    PPOHParams,
    PPOState,
    TEnvParams,
    TEnvState,
    Trajectory,
    TrajectoryWithLastObs,
    _actor,
    render_episode,
)
from .testsuites.algorithm_tests import LearningAlgorithmTests

logger = getLogger(__name__)


@pytest.fixture(scope="session", params=[123])
def seed(request: pytest.FixtureRequest) -> int:
    seed = getattr(request, "param", 123)
    return seed


@pytest.fixture(scope="session")
def rng(seed: int) -> chex.PRNGKey:
    return jax.random.key(seed)


@pytest.fixture(scope="session")
def n_agents(request: pytest.FixtureRequest) -> int | None:
    return getattr(request, "param", None)


@pytest.fixture(scope="session")
def results_ours(
    algo: JaxRLExample,
    rng: chex.PRNGKey,
    n_agents: int | None,
):
    train_fn = algo.train

    if n_agents is not None:
        train_fn = jax.vmap(train_fn)
        rng = jax.random.split(rng, n_agents)

    train_fn = jax.jit(train_fn).lower(rng).compile()
    _start = time.perf_counter()
    train_states_ours, evals_ours = train_fn(rng)
    jax.block_until_ready((train_states_ours, evals_ours))
    print(f"Our tweaked rejax.PPO: {time.perf_counter() - _start:.1f} seconds.")
    return train_states_ours, evals_ours


@pytest.fixture
def results_ours_with_trainer(
    algo: JaxRLExample,
    rng: chex.PRNGKey,
    n_agents: int,
    jax_trainer: JaxTrainer,
):
    train_fn = jax_trainer.fit

    if n_agents is not None:
        jax_trainer = jax_trainer.replace(callbacks=())
        train_fn = jax_trainer.fit
        train_fn = jax.vmap(train_fn, in_axes=(None, 0))
        rng = jax.random.split(rng, n_agents)

    train_fn_with_trainer = jax.jit(train_fn).lower(algo, rng).compile()
    _start = time.perf_counter()
    _train_states_ours_with_trainer, evals_ours_with_trainer = train_fn_with_trainer(algo, rng)
    jax.block_until_ready((_train_states_ours_with_trainer, evals_ours_with_trainer))
    print(f"Our tweaked rejax.PPO with JaxTrainer: {time.perf_counter() - _start:.1f} seconds.")
    return _train_states_ours_with_trainer, evals_ours_with_trainer


@pytest.fixture
def results_rejax(
    algo: JaxRLExample,
    rng: chex.PRNGKey,
    n_agents: int,
):
    # _start = time.perf_counter()
    _rejax_ppo, train_states_rejax, evals_rejax = _train_rejax(
        env=algo.env, env_params=algo.env_params, hp=algo.hp, rng=rng, n_agents=n_agents
    )
    # jax.block_until_ready((train_states_rejax, evals_rejax))
    # print(f"rejax.PPO: {time.perf_counter() - _start:.1f} seconds.")
    return _rejax_ppo, train_states_rejax, evals_rejax


def test_ours(
    algo: JaxRLExample,
    results_ours: tuple[PPOState, EvalMetrics],
    tensor_regression: TensorRegressionFixture,
    seed: int,
    rng: chex.PRNGKey,
    n_agents: int | None,
    original_datadir: Path,
):
    evaluations = results_ours[1]
    tensor_regression.check(jax.tree.map(lambda v: v.__array__(), dataclasses.asdict(evaluations)))

    eval_rng = rng
    if n_agents is None:
        gif_path = original_datadir / f"ours_{seed=}.gif"
        algo.visualize(results_ours[0], gif_path=gif_path, eval_rng=eval_rng)
    else:
        gif_path = original_datadir / f"ours_{n_agents=}_{seed=}_first.gif"
        fn = functools.partial(jax.tree.map, operator.itemgetter(0))
        algo.visualize(fn(results_ours[0]), gif_path=gif_path, eval_rng=eval_rng)


def test_ours_with_trainer(
    algo: JaxRLExample,
    results_ours_with_trainer: tuple[PPOState, EvalMetrics],
    tensor_regression: TensorRegressionFixture,
    tmp_path: Path,
    seed: int,
    rng: chex.PRNGKey,
    n_agents: int | None,
    original_datadir: Path,
):
    ts, evaluations = results_ours_with_trainer
    tensor_regression.check(jax.tree.map(lambda v: v.__array__(), dataclasses.asdict(evaluations)))

    eval_rng = rng
    if n_agents is None:
        gif_path = original_datadir / f"ours_with_trainer_{seed=}.gif"
        algo.visualize(ts, gif_path=gif_path, eval_rng=eval_rng)
    else:
        gif_path = original_datadir / f"ours_with_trainer_{n_agents=}_{seed=}_first.gif"
        fn = functools.partial(jax.tree.map, operator.itemgetter(0))
        algo.visualize(fn(ts), gif_path=gif_path, eval_rng=eval_rng)


def test_results_are_same_with_or_without_jax_trainer(
    results_ours: tuple[PPOState, EvalMetrics],
    results_ours_with_trainer: tuple[PPOState, EvalMetrics],
):
    np.testing.assert_allclose(
        results_ours[1].cumulative_reward, results_ours_with_trainer[1].cumulative_reward
    )
    # jax.tree.map(
    #     np.testing.assert_allclose,
    #     jax.tree.leaves(results_ours),
    #     jax.tree.leaves(results_ours_with_trainer),
    # )


def test_rejax(
    rng: chex.PRNGKey,
    results_rejax: tuple[rejax.PPO, Any, EvalMetrics],
    tensor_regression: TensorRegressionFixture,
    original_datadir: Path,
    n_agents: int | None,
):
    """Train `rejax.PPO` with the same parameters."""

    _algo, ts, evaluations = results_rejax
    tensor_regression.check(jax.tree.map(lambda v: v.__array__(), dataclasses.asdict(evaluations)))
    eval_rng = rng

    if n_agents is None:
        gif_path = original_datadir / f"rejax_{seed=}.gif"
        _visualize_rejax(rejax_algo=_algo, rejax_ts=ts, eval_rng=rng, gif_path=gif_path)
    else:
        fn = functools.partial(jax.tree.map, operator.itemgetter(0))
        _visualize_rejax(
            rejax_algo=results_rejax[0],
            rejax_ts=fn(results_rejax[1]),
            eval_rng=eval_rng,
            gif_path=original_datadir / f"rejax_{n_agents=}_{seed=}_first.gif.gif",
        )


def best_index(evals: EvalMetrics) -> int:
    return jnp.argmax(evals.cumulative_reward[:, -1]).item()


def median_index(evals: EvalMetrics) -> int:
    vals = evals.cumulative_reward[:, -1]
    return jnp.argsort(vals)[len(vals) // 2].item()


def get_slicing_fn(eval: EvalMetrics, get_index_fn: Callable[[EvalMetrics], int]) -> Any:
    index = get_index_fn(eval)
    return functools.partial(jax.tree.map, operator.itemgetter(index))


@pytest.mark.parametrize("n_agents", [pytest.param(100, marks=pytest.mark.slow)], indirect=True)
def test_algos_are_equivalent(
    algo: JaxRLExample,
    n_agents: int | None,
    results_ours: tuple[PPOState, EvalMetrics],
    results_rejax: tuple[rejax.PPO, Any, EvalMetrics],
):
    if n_agents is None:
        _ours_vs_rejax = scipy.stats.mannwhitneyu(
            results_ours[1].cumulative_reward[-1], results_rejax[2].cumulative_reward[-1]
        )
    else:
        _ours_vs_rejax = scipy.stats.mannwhitneyu(
            results_ours[1].cumulative_reward[:, -1], results_rejax[2].cumulative_reward[:, -1]
        )
    # TODO: interpret these results.


def _visualize_rejax(rejax_algo: rejax.PPO, rejax_ts: Any, eval_rng: chex.PRNGKey, gif_path: Path):
    # rejax_algo = results_rejax[0]
    # ts = results_rejax[1]
    actor_ts = rejax_ts.actor_ts.replace(apply_fn=rejax_algo.actor.apply)
    actor = functools.partial(
        _actor,
        actor_ts=actor_ts,
        rms_state=rejax_ts.rms_state,
        normalize_observations=rejax_algo.normalize_observations,
    )
    render_episode(
        actor=actor,
        env=rejax_algo.env,
        env_params=rejax_algo.env_params,
        gif_path=gif_path,
        rng=eval_rng,
    )


def _train_rejax(
    env: Environment[gymnax.EnvState, TEnvParams],
    env_params: TEnvParams,
    hp: PPOHParams,
    rng: chex.PRNGKey,
    n_agents: int | None = None,
):
    print("Rejax")
    # todo: Make sure that rejax uses the same number of epochs as us.
    algo = rejax.PPO.create(
        env=env,
        env_params=env_params,
        num_envs=hp.num_envs,  # =100,
        num_steps=hp.num_steps,  # =100,
        num_epochs=hp.num_epochs,  # =10,
        num_minibatches=hp.num_minibatches,  # =10,
        learning_rate=hp.learning_rate,  # =0.001,
        max_grad_norm=hp.max_grad_norm,  # =10,
        total_timesteps=hp.total_timesteps,  # =150_000,
        eval_freq=hp.eval_freq,  # =2000,
        gamma=hp.gamma,  # =0.995,
        gae_lambda=hp.gae_lambda,  # =0.95,
        clip_eps=hp.clip_eps,  # =0.2,
        ent_coef=hp.ent_coef,  # =0.0,
        vf_coef=hp.vf_coef,  # =0.5,
        normalize_observations=hp.normalize_observations,  # =True,
    )
    print("Compiling...")
    start = time.perf_counter()

    train_fn = algo.train
    if n_agents:
        # Vmap training function over n_agents initial seeds
        train_fn = jax.vmap(train_fn)
        rng = jax.random.split(rng, n_agents)

    train_fn = jax.jit(train_fn).lower(rng).compile()
    print(f"Compiled in {time.perf_counter() - start} seconds.")
    print("Training...")
    start = time.perf_counter()
    ts, eval = train_fn(rng)
    jax.block_until_ready((ts, eval))
    print(f"Finished training in {time.perf_counter() - start} seconds.")
    return algo, ts, EvalMetrics(eval[0], eval[1])

    # print(ts)


def train_lightning(
    algo: JaxRLExample,
    rng: chex.PRNGKey,
    trainer: lightning.Trainer,
):
    # Fit with pytorch-lightning.
    print("Lightning")

    module = PPOLightningModule(
        learner=algo,
        ts=algo.init_train_state(rng),
    )

    start = time.perf_counter()
    trainer.fit(module)
    print(f"Trained in {time.perf_counter() - start:.1f} seconds.")

    evaluation = trainer.validate(module)

    return module.ts, evaluation


@pytest.fixture(scope="session", params=["Pendulum-v1"])
def env_id(request: pytest.FixtureRequest) -> str:
    # env_id = "halfcheetah"
    # env_id = "humanoid"
    return request.param


@pytest.fixture(scope="session")
def env_and_params(env_id: str) -> tuple[Environment[gymnax.EnvState, TEnvParams], TEnvParams]:
    from brax.envs import _envs as brax_envs
    from rejax.compat.brax2gymnax import create_brax

    env: Environment[gymnax.EnvState, gymnax.EnvParams]
    env_params: gymnax.EnvParams
    if env_id in brax_envs:
        env, env_params = create_brax(  # type: ignore
            env_id,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
            batch_size=None,
            backend="generalized",
        )
    elif isinstance(env_id, str):
        env, env_params = gymnax.make(env_id=env_id)  # type: ignore
    else:
        env = env_id()  # type: ignore
        env_params = env.default_params
    return env, env_params  # type: ignore


@pytest.fixture(scope="session")
def algo(
    env_and_params: tuple[Environment[TEnvState, TEnvParams], TEnvParams],
) -> JaxRLExample[TEnvState, TEnvParams]:
    env, env_params = env_and_params
    algo = JaxRLExample[TEnvState, TEnvParams](
        env=env,
        env_params=env_params,
        actor=JaxRLExample.create_actor(env, env_params),
        critic=JaxRLExample.create_critic(),
        hp=PPOHParams(
            num_envs=100,
            num_steps=100,
            num_epochs=10,
            num_minibatches=10,
            learning_rate=0.001,
            max_grad_norm=10,
            total_timesteps=150_000,
            eval_freq=2000,
            gamma=0.995,
            gae_lambda=0.95,
            clip_eps=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            normalize_observations=True,
            debug=False,
        ),
    )
    return algo


@pytest.fixture(autouse=True, scope="session")
def debug_jit_warnings():
    # Temporarily make this particular warning into an error to help future-proof our jax code.
    import jax._src.deprecations

    val_before = jax._src.deprecations._registered_deprecations["tracer-hash"].accelerated
    jax._src.deprecations._registered_deprecations["tracer-hash"].accelerated = True
    yield
    jax._src.deprecations._registered_deprecations["tracer-hash"].accelerated = val_before

    # train_pure_jax(algo, backend="cpu")
    # train_rejax(env=algo.env, env_params=algo.env_params, hp=algo.hp, backend="cpu")
    # train_lightning(algo, accelerator="cpu")


@pytest.fixture
def max_epochs(algo: JaxRLExample, request: pytest.FixtureRequest) -> int:
    # This is the usual value: (75)
    # return 3  # shorter for tests?
    default_max_epochs = int(np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int))
    return getattr(request, "param", default_max_epochs)


@pytest.fixture
def jax_trainer(algo: JaxRLExample, max_epochs: int, tmp_path: Path):
    iteration_steps = algo.hp.num_envs * algo.hp.num_steps
    num_iterations = np.ceil(algo.hp.eval_freq / iteration_steps).astype(int)
    training_steps_per_epoch: int = num_iterations

    return JaxTrainer(
        max_epochs=max_epochs,
        training_steps_per_epoch=training_steps_per_epoch,
        # todo: make sure that this also works with the wandb logger!
        logger=CSVLogger(save_dir=tmp_path, name=None, flush_logs_every_n_steps=1),
        default_root_dir=tmp_path,
        callbacks=(
            # RlThroughputCallback(),  # Can't use this callback with `vmap`!
            # RenderEpisodesCallback(on_every_epoch=False),
            RichProgressBar(),
        ),
    )


## Pytorch-Lightning wrapper around this learner:

# Don't allow tests to run for more than 5 seconds.
# pytestmark = pytest.mark.timeout(5)


class PPOLightningModule(lightning.LightningModule):
    """Uses the same code as [project.algorithms.jax_rl_example.JaxRLExample][], but the training
    loop is run with pytorch-lightning.

    This is currently only meant to be used to compare the difference fully-jitted training loop
    and lightning.
    """

    def __init__(
        self,
        learner: JaxRLExample,
        ts: PPOState,
    ):
        # https://github.com/keraJLi/rejax/blob/a1428ad3d661e31985c5c19460cec70bc95aef6e/configs/gymnax/pendulum.yaml#L1

        super().__init__()
        self.learner = learner
        self.ts = ts

        self.save_hyperparameters(hparams_to_dict(learner))
        self.actor_params = torch.nn.ParameterList(
            jax.tree.leaves(
                jax.tree.map(
                    torch_jax_interop.to_torch.jax_to_torch_tensor,
                    self.ts.actor_ts.params,
                )
            )
        )
        self.critic_params = torch.nn.ParameterList(
            jax.tree.leaves(
                jax.tree.map(
                    torch_jax_interop.to_torch.jax_to_torch_tensor,
                    self.ts.critic_ts.params,
                )
            )
        )

        self.automatic_optimization = False

        iteration_steps = self.learner.hp.num_envs * self.learner.hp.num_steps
        # number of "iterations" (collecting batches of episodes in the environment) per epoch.
        self.num_train_iterations = np.ceil(self.learner.hp.eval_freq / iteration_steps).astype(
            int
        )

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        start = time.perf_counter()
        with jax.disable_jit(self.learner.hp.debug):
            algo_struct = self.learner
            self.ts, train_metrics = algo_struct.fused_training_step(batch_idx, self.ts)

        duration = time.perf_counter() - start
        logger.debug(f"Training step took {duration:.1f} seconds.")
        actor_losses = train_metrics.actor_losses
        critic_losses = train_metrics.critic_losses
        self.log("train/actor_loss", actor_losses.mean().item(), logger=True, prog_bar=True)
        self.log("train/critic_loss", critic_losses.mean().item(), logger=True, prog_bar=True)

        updates_per_second = (
            self.learner.hp.num_epochs * self.learner.hp.num_minibatches
        ) / duration
        self.log("train/updates_per_second", updates_per_second, logger=True, prog_bar=True)
        minibatch_size = (
            self.learner.hp.num_envs * self.learner.hp.num_steps
        ) // self.learner.hp.num_minibatches
        samples_per_update = minibatch_size
        self.log(
            "train/samples_per_second",
            updates_per_second * samples_per_update,
            logger=True,
            prog_bar=True,
            on_step=True,
        )

        # for jax_param, torch_param in zip(
        #     jax.tree.leaves(self.train_state.actor_ts.params), self.actor_params
        # ):
        #     torch_param.set_(torch_jax_interop.to_torch.jax_to_torch_tensor(jax_param))

        # for jax_param, torch_param in zip(
        #     jax.tree.leaves(self.train_state.critic_ts.params), self.critic_params
        # ):
        #     torch_param.set_(torch_jax_interop.to_torch.jax_to_torch_tensor(jax_param))

        return

    @override
    def train_dataloader(self) -> Iterable[Trajectory]:
        # BUG: what's probably happening is that the dataloader keeps getting batches with the
        # initial train state!
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.arange(self.num_train_iterations, device=self.device))
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False, collate_fn=None)

    def val_dataloader(self) -> Any:
        # todo: unsure what this should be yielding..
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.arange(1, device=self.device))
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False, collate_fn=None)

    def validation_step(self, batch: int, batch_index: int):
        # self.learner.eval_callback()
        # return  # skip the rest for now while we compare the performance?
        eval_metrics = self.learner.eval_callback(ts=self.ts)
        episode_lengths = eval_metrics.episode_length
        cumulative_rewards = eval_metrics.cumulative_reward
        self.log("val/episode_lengths", episode_lengths.mean().item(), batch_size=1)
        self.log("val/rewards", cumulative_rewards.mean().item(), batch_size=1)

    @override
    def configure_optimizers(self) -> Any:
        # todo: Note, this one isn't used atm!
        from torch.optim.adam import Adam

        return Adam(self.parameters(), lr=1e-3)

    @override
    def transfer_batch_to_device(
        self, batch: TrajectoryWithLastObs | int, device: torch.device, dataloader_idx: int
    ) -> TrajectoryWithLastObs | int:
        if isinstance(batch, int):
            # FIXME: valid dataloader currently just yields ints, not trajectories.
            return batch
        if isinstance(batch, list) and len(batch) == 1:
            # FIXME: train dataloader currently just yields ints, not trajectories.
            return batch

        _batch_jax_devices = batch.trajectories.obs.devices()
        assert len(_batch_jax_devices) == 1
        batch_jax_device = _batch_jax_devices.pop()
        torch_self_device = device
        if (
            torch_self_device.type == "cuda"
            and "cuda" in str(batch_jax_device)
            and (torch_self_device.index == -1 or torch_self_device.index == batch_jax_device.id)
        ):
            # All good, both are on the same GPU.
            return batch

        jax_self_device = torch_jax_interop.to_jax.torch_to_jax_device(torch_self_device)
        return jax.tree.map(functools.partial(jax.device_put, device=jax_self_device), batch)


class RlThroughputCallback(MeasureSamplesPerSecondCallback):
    """A callback to measure the throughput of RL algorithms."""

    def __init__(self, num_optimizers: int | None = 1):
        super().__init__(num_optimizers=num_optimizers)
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()
        self._updates = 0

    @override
    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        super().on_fit_start(trainer, pl_module)
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()

    @override
    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: TrajectoryWithLastObs,
        batch_index: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_index)
        if not isinstance(batch, TrajectoryWithLastObs):
            return
        episodes = batch.trajectories
        assert episodes.obs.shape
        num_episodes = episodes.obs.shape[0]
        num_transitions = np.prod(episodes.obs.shape[:2])
        self.total_episodes += num_episodes
        self.total_transitions += num_transitions
        steps_per_second = self.total_transitions / (time.perf_counter() - self._start)
        updates_per_second = (self._updates) / (time.perf_counter() - self._start)
        episodes_per_second = self.total_episodes / (time.perf_counter() - self._start)
        logger.info(
            f"Total transitions: {self.total_transitions}, total episodes: {self.total_episodes}"
        )
        # print(f"Steps per second: {steps_per_second}")
        logger.info(f"Steps per second: {steps_per_second}")
        logger.info(f"Episodes per second: {episodes_per_second}")
        logger.info(f"Updates per second: {updates_per_second}")

    @override
    def get_num_samples(self, batch: TrajectoryWithLastObs) -> int:
        if isinstance(batch, int):  # fixme
            return 1
        return int(np.prod(batch.trajectories.obs.shape[:2]).item())

    @override
    def on_fit_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        super().on_fit_end(trainer, pl_module)

    @override
    def log(
        self,
        name: str,
        value: Any,
        module: JaxRLExample,
        trainer: lightning.Trainer | JaxTrainer,
        **kwargs,
    ):
        # Used to possibly customize how the values are logged (e.g. for non-LightningModules).
        # By default, uses the LightningModule.log method.
        # TODO: Somehow log the metrics without an actual trainer?
        # Should we create a Trainer / LightningModule "facade" that the callbacks can interact with?
        if trainer.logger:
            trainer.logger.log_metrics({name: value}, step=trainer.global_step, **kwargs)

        # if trainer.progress_bar_callback:
        #     trainer.progress_bar_callback.log_metrics({name: value}, step=trainer.global_step, **kwargs)
        # return trainer.logger.log_metrics().log(
        #     name,
        #     value,
        #     **kwargs,
        # )


# TODO: potentially just use the Lightning adapter for unit tests for now?
@pytest.mark.skip(reason="TODO: ests assume a LightningModule atm (.state_dict()), etc.")
@run_for_all_configs_of_type("algorithm", JaxRLExample)
class TestJaxRLExample(LearningAlgorithmTests[JaxRLExample]):  # type: ignore
    pass


@pytest.fixture
def lightning_trainer(max_epochs: int, tmp_path: Path):
    return lightning.Trainer(
        max_epochs=max_epochs,
        # logger=CSVLogger(save_dir="logs/jax_rl_debug"),
        logger=False,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        default_root_dir=tmp_path,
        # barebones=True,
        # reload_dataloaders_every_n_epochs=1,  # todo: use this if we end up making a generator in train_dataloader
        enable_checkpointing=False,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=max_epochs,
        # limit_val_batches=0,
        fast_dev_run=False,
        detect_anomaly=False,
        profiler=None,
    )


# reducing the max_epochs from 75 down to 10 because it's just wayyy too slow.
@pytest.mark.xfail(reason="Seems to not be completely reproducible")
@pytest.mark.slow
# @pytest.mark.timeout(80)
@pytest.mark.parametrize("max_epochs", [15], indirect=True)
def test_lightning(
    algo: JaxRLExample,
    rng: chex.PRNGKey,
    lightning_trainer: lightning.Trainer,
    tensor_regression: TensorRegressionFixture,
    original_datadir: Path,
):
    # todo: save a gif and some metrics?
    train_state, evaluations = train_lightning(
        algo,
        rng=rng,
        trainer=lightning_trainer,
    )
    gif_path = original_datadir / "lightning.gif"
    algo.visualize(train_state, gif_path=gif_path)
    # file_regression.check(gif_path.read_bytes(), binary=True, extension=".gif")
    assert len(evaluations) == 1
    # floats in regression files are saved with full precision, and the last few digits are
    # different for some reason.
    tensor_regression.check(jax.tree.map(np.asarray, evaluations[0]))

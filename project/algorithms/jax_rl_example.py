from __future__ import annotations

import contextlib
import functools
import operator
import time
from collections.abc import Callable, Iterable, Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypedDict

import chex
import flax.core
import flax.linen
import flax.struct
import gymnax
import gymnax.environments.spaces
import gymnax.experimental.rollout
import jax
import jax.experimental
import jax.numpy as jnp
import lightning
import lightning.pytorch
import lightning.pytorch.callbacks
import lightning.pytorch.callbacks.progress
import lightning.pytorch.callbacks.progress.rich_progress
import lightning.pytorch.loggers
import lightning.pytorch.trainer
import lightning.pytorch.trainer.states
import numpy as np
import optax
import rejax
import rejax.evaluate
import torch
import torch_jax_interop
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment
from gymnax.visualize.visualizer import Visualizer
from jax._src.sharding_impls import UNSPECIFIED, Device
from matplotlib import pyplot as plt
from rejax.algos.mixins import RMSState
from rejax.evaluate import evaluate
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork
from torch.utils.data import DataLoader
from typing_extensions import TypeVar, override
from xtils.jitpp import Static

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.algorithms.jax_trainer import JaxCallback, JaxModule, JaxTrainer, hparams_to_dict
from project.utils.env_vars import REPO_ROOTDIR

logger = get_logger(__name__)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_EnvParams = TypeVar("_EnvParams", bound=gymnax.EnvParams, default=gymnax.EnvParams)
_EnvState = TypeVar("_EnvState", bound=gymnax.EnvState, default=gymnax.EnvState)


class Trajectory(flax.struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    value: jax.Array
    done: jax.Array


class TrajectoryWithLastObs(flax.struct.PyTreeNode):
    trajectories: Trajectory
    last_done: jax.Array
    last_obs: jax.Array


class AdvantageMinibatch(flax.struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class TrajectoryCollectionState(Generic[_EnvState], flax.struct.PyTreeNode):
    last_obs: jax.Array
    env_state: _EnvState
    rms_state: RMSState
    last_done: jax.Array
    global_step: int
    rng: chex.PRNGKey


class PPOState(Generic[_EnvState], flax.struct.PyTreeNode):
    actor_ts: TrainState
    critic_ts: TrainState
    rng: chex.PRNGKey
    data_collection_state: TrajectoryCollectionState[_EnvState]


class PPOHParams(flax.struct.PyTreeNode):
    """Hyper-parameters for this PPO example.

    These are taken from `rejax.PPO` algorithm class.
    """

    num_epochs: int = flax.struct.field(pytree_node=False, default=8)
    num_envs: int = flax.struct.field(pytree_node=False, default=64)  # overwrite default
    num_steps: int = flax.struct.field(pytree_node=False, default=64)
    num_minibatches: int = flax.struct.field(pytree_node=False, default=16)

    eval_freq: int = flax.struct.field(pytree_node=False, default=4_096)

    normalize_observations: bool = flax.struct.field(pytree_node=False, default=False)
    total_timesteps: int = flax.struct.field(pytree_node=False, default=131_072)
    debug: bool = flax.struct.field(pytree_node=False, default=False)

    learning_rate: chex.Scalar = flax.struct.field(pytree_node=True, default=0.0003)
    gamma: chex.Scalar = flax.struct.field(pytree_node=True, default=0.99)
    max_grad_norm: chex.Scalar = flax.struct.field(pytree_node=True, default=jnp.inf)

    gae_lambda: chex.Scalar = flax.struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = flax.struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = flax.struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = flax.struct.field(pytree_node=True, default=0.01)

    # rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(0))
    # networks_rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(1))
    # env_rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(2))


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


class _AgentKwargs(TypedDict):
    activation: str
    hidden_layer_sizes: Sequence[int]


class _NetworkConfig(TypedDict):
    agent_kwargs: _AgentKwargs


class TrainStepMetrics(TypedDict):
    actor_losses: jax.Array
    critic_losses: jax.Array


class EvalMetrics(flax.struct.PyTreeNode):
    episode_length: jax.Array
    cumulative_reward: jax.Array


class PPOLearner(
    flax.struct.PyTreeNode,
    JaxModule[PPOState[_EnvState], TrajectoryWithLastObs, EvalMetrics],
    Generic[_EnvState, _EnvParams],
):
    """PPO algorithm based on `rejax.PPO`.

    Differences w.r.t. rejax.PPO:
    - The state / hparams are split into different, fully-typed structs:
        - The algorithm state is in a typed `PPOState` struct (vs an untyped,
            dynamically-generated struct in rejax).
        - The hyper-parameters are in a typed `PPOHParams` struct.
        - The state variables related to the collection of data from the environment is a
            `TrajectoryCollectionState` instead of everything together.
    - The seeds for the networks and the environment data collection are separated.

    The logic is exactly the same: The losses / updates are computed in the exact same way.
    """

    env: Environment[_EnvState, _EnvParams] = flax.struct.field(pytree_node=False)
    env_params: _EnvParams
    actor: flax.linen.Module = flax.struct.field(pytree_node=False)
    critic: flax.linen.Module = flax.struct.field(pytree_node=False)
    hp: PPOHParams = flax.struct.field(pytree_node=True)

    HParams = PPOHParams

    @classmethod
    def create(
        cls,
        env_id: str | None = None,
        env: Environment[_EnvState, _EnvParams] | None = None,
        env_params: _EnvParams | None = None,
        hp: PPOHParams | None = None,
    ):
        from brax.envs import _envs as brax_envs
        from rejax.compat.brax2gymnax import create_brax

        # env_params: gymnax.EnvParams
        if env_id is None:
            assert env is not None
            env_params = env_params or env.default_params  # type: ignore
        elif env_id in brax_envs:
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
            raise NotImplementedError(env_id)

        assert env is not None
        assert env_params is not None
        return cls(
            env=env,
            env_params=env_params,
            actor=cls.create_actor(env, env_params),
            critic=cls.create_critic(),
            hp=hp or cls.HParams(),
        )

    @classmethod
    def create_networks(
        cls,
        env: Environment[gymnax.EnvState, _EnvParams],
        env_params: _EnvParams,
        config: _NetworkConfig,
    ):
        # Equivalent to:
        # return rejax.PPO.create_agent(config, env, env_params)
        return {
            "actor": cls.create_actor(env, env_params, **config["agent_kwargs"]),
            "critic": cls.create_actor(env, env_params, **config["agent_kwargs"]),
        }

    @classmethod
    def create_actor(
        cls,
        env: Environment[Any, _EnvParams],
        env_params: _EnvParams,
        activation: str | Callable[[jax.Array], jax.Array] = "swish",
        hidden_layer_sizes: Sequence[int] = (64, 64),
        **actor_kwargs,
    ) -> DiscretePolicy | GaussianPolicy:
        activation_fn: Callable[[jax.Array], jax.Array] = (
            getattr(flax.linen, activation) if not callable(activation) else activation
        )
        hidden_layer_sizes = tuple(hidden_layer_sizes)
        action_space = env.action_space(env_params)

        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            return DiscretePolicy(
                action_space.n,
                activation=activation_fn,
                hidden_layer_sizes=hidden_layer_sizes,
                **actor_kwargs,
            )
        assert isinstance(action_space, gymnax.environments.spaces.Box)
        return GaussianPolicy(
            np.prod(action_space.shape),
            (action_space.low, action_space.high),  # type: ignore
            activation=activation_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            **actor_kwargs,
        )

    @classmethod
    def create_critic(
        cls,
        activation: str | Callable[[jax.Array], jax.Array] = "swish",
        hidden_layer_sizes: Sequence[int] = (64, 64),
        **critic_kwargs,
    ) -> VNetwork:
        activation_fn: Callable[[jax.Array], jax.Array] = (
            getattr(flax.linen, activation) if isinstance(activation, str) else activation
        )
        hidden_layer_sizes = tuple(hidden_layer_sizes)
        return VNetwork(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation_fn, **critic_kwargs
        )

    def init_train_state(self, rng: chex.PRNGKey) -> PPOState[_EnvState]:
        rng, networks_rng, env_rng = jax.random.split(rng, 3)

        rng_actor, rng_critic = jax.random.split(networks_rng, 2)

        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])

        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        critic_params = self.critic.init(rng_critic, obs_ph)

        tx = optax.adam(learning_rate=self.hp.learning_rate)
        # TODO: Why isn't the `apply_fn` not set in rejax?
        actor_ts = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=self.critic.apply, params=critic_params, tx=tx)

        env_rng, reset_rng = jax.random.split(env_rng)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            jax.random.split(reset_rng, self.hp.num_envs), self.env_params
        )

        collection_state = TrajectoryCollectionState(
            last_obs=obs,
            rms_state=RMSState.create(shape=obs_ph.shape),
            global_step=0,
            env_state=env_state,
            last_done=jnp.zeros(self.hp.num_envs, dtype=bool),
            rng=env_rng,
        )

        return PPOState(
            actor_ts=actor_ts,
            critic_ts=critic_ts,
            rng=rng,
            data_collection_state=collection_state,
        )

    @functools.partial(jit, static_argnames=["skip_initial_evaluation"])
    def train(
        self,
        rng: jax.Array,
        train_state: PPOState[_EnvState] | None = None,
        skip_initial_evaluation: bool = False,
    ) -> tuple[PPOState[_EnvState], EvalMetrics]:
        """Full training loop in pure jax (a lot faster than when using pytorch-lightning).

        Unfolded version of `rejax.PPO.train`.

        Training loop in pure jax (a lot faster than when using pytorch-lightning).
        """
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state if train_state is not None else self.init_train_state(rng)

        initial_evaluation: EvalMetrics | None = None
        if not skip_initial_evaluation:
            initial_evaluation = self.eval_callback(ts)

        num_evals = np.ceil(self.hp.total_timesteps / self.hp.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(
            self.training_epoch,
            init=ts,
            xs=None,
            length=num_evals,
        )

        if not skip_initial_evaluation:
            assert initial_evaluation is not None
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )
            assert isinstance(evaluation, EvalMetrics)

        return ts, evaluation

    @jit
    def training_epoch(
        self, ts: PPOState[_EnvState], epoch: int
    ) -> tuple[PPOState[_EnvState], EvalMetrics]:
        # Run a few training iterations
        iteration_steps = self.hp.num_envs * self.hp.num_steps
        num_iterations = np.ceil(self.hp.eval_freq / iteration_steps).astype(int)
        ts = jax.lax.fori_loop(
            0,
            num_iterations,
            # drop metrics for now
            lambda i, train_state_i: self.fused_training_step(i, train_state_i)[0],
            ts,
        )
        # Run evaluation
        return ts, self.eval_callback(ts, ts.rng)

    @jit
    def fused_training_step(self, iteration: int, ts: PPOState[_EnvState]):
        """Fused training step in jax (joined data collection + training).

        *MUCH* faster than using pytorch-lightning, but you lose the callbacks and such.
        """

        data_collection_state, trajectories = self.collect_trajectories(
            # env=self.env,
            # env_params=self.env_params,
            # actor=self.actor,
            # critic=self.critic,
            collection_state=ts.data_collection_state,
            actor_params=ts.actor_ts.params,
            critic_params=ts.critic_ts.params,
            # num_envs=self.hp.num_envs,
            # num_steps=self.hp.num_steps,
            # discrete=discrete,
            # normalize_observations=self.hp.normalize_observations,
        )
        ts = ts.replace(data_collection_state=data_collection_state)
        return self.training_step(iteration, ts, trajectories)

    @jit
    def training_step(self, batch_idx: int, ts: PPOState[_EnvState], batch: TrajectoryWithLastObs):
        """Training step in pure jax."""
        trajectories = batch

        ts, (actor_losses, critic_losses) = jax.lax.scan(
            functools.partial(self.ppo_update_epoch, trajectories=trajectories),
            init=ts,
            xs=jnp.arange(self.hp.num_epochs),  # type: ignore
            length=self.hp.num_epochs,
        )
        # todo: perhaps we could have a callback that updates a progress bar?
        # jax.debug.print("actor_losses {}: {}", iteration, actor_losses.mean())
        # jax.debug.print("critic_losses {}: {}", iteration, critic_losses.mean())

        return ts, TrainStepMetrics(actor_losses=actor_losses, critic_losses=critic_losses)

    @jit
    def ppo_update_epoch(
        self, ts: PPOState[_EnvState], epoch_index: int, trajectories: TrajectoryWithLastObs
    ):
        minibatch_rng = jax.random.fold_in(ts.rng, epoch_index)

        last_val = self.critic.apply(ts.critic_ts.params, ts.data_collection_state.last_obs)
        assert isinstance(last_val, jax.Array)
        last_val = jnp.where(ts.data_collection_state.last_done, 0, last_val)
        advantages, targets = calculate_gae(
            trajectories, last_val, gamma=self.hp.gamma, gae_lambda=self.hp.gae_lambda
        )
        batch = AdvantageMinibatch(trajectories.trajectories, advantages, targets)
        minibatches = shuffle_and_split(
            batch, minibatch_rng, num_minibatches=self.hp.num_minibatches
        )

        # shuffle the data and split it into minibatches

        num_steps = self.hp.num_steps
        num_envs = self.hp.num_envs
        num_minibatches = self.hp.num_minibatches
        assert (num_envs * num_steps) % num_minibatches == 0
        minibatches = shuffle_and_split(
            batch,
            minibatch_rng,
            num_minibatches=num_minibatches,
        )
        return jax.lax.scan(self.ppo_update, ts, minibatches, length=self.hp.num_minibatches)

    @jit
    def ppo_update(self, ts: PPOState[_EnvState], batch: AdvantageMinibatch):
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
            ts.actor_ts.params,
            actor=self.actor,
            batch=batch,
            clip_eps=self.hp.clip_eps,
            ent_coef=self.hp.ent_coef,
        )
        assert isinstance(actor_loss, jax.Array)
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            ts.critic_ts.params,
            critic=self.critic,
            batch=batch,
            clip_eps=self.hp.clip_eps,
            vf_coef=self.hp.vf_coef,
        )
        assert isinstance(critic_loss, jax.Array)

        # TODO: to log the loss here?
        actor_ts = ts.actor_ts.apply_gradients(grads=actor_grads)
        critic_ts = ts.critic_ts.apply_gradients(grads=critic_grads)

        return ts.replace(actor_ts=actor_ts, critic_ts=critic_ts), (actor_loss, critic_loss)

    def eval_callback(self, ts: PPOState[_EnvState]) -> EvalMetrics:
        actor = make_actor(ts=ts, hp=self.hp)
        max_steps = self.env_params.max_steps_in_episode
        ep_lengths, cum_rewards = evaluate(
            actor, ts.rng, self.env, self.env_params, 128, max_steps
        )
        return EvalMetrics(episode_length=ep_lengths, cumulative_reward=cum_rewards)

    def get_batch(
        self, ts: PPOState[_EnvState], batch_idx: int
    ) -> tuple[PPOState[_EnvState], TrajectoryWithLastObs]:
        data_collection_state, trajectories = self.collect_trajectories(
            ts.data_collection_state,
            actor_params=ts.actor_ts.params,
            critic_params=ts.critic_ts.params,
        )
        ts = ts.replace(data_collection_state=data_collection_state)
        return ts, trajectories

    @jit
    def collect_trajectories(
        self,
        collection_state: TrajectoryCollectionState[_EnvState],
        actor_params: FrozenVariableDict,
        critic_params: FrozenVariableDict,
    ):
        env_step_fn = functools.partial(
            self.env_step,
            # env=self.env,
            # env_params=self.env_params,
            # actor=self.actor,
            # critic=self.critic,
            # num_envs=self.hp.num_envs,
            actor_params=actor_params,
            critic_params=critic_params,
            # discrete=self.discrete,
            # normalize_observations=self.hp.normalize_observations,
        )
        collection_state, trajectories = jax.lax.scan(
            env_step_fn,
            collection_state,
            xs=jnp.arange(self.hp.num_steps),
            length=self.hp.num_steps,
        )
        trajectories_with_last = TrajectoryWithLastObs(
            trajectories=trajectories,
            last_done=collection_state.last_done,
            last_obs=collection_state.last_obs,
        )
        return collection_state, trajectories_with_last

    @jit
    def env_step(
        self,
        collection_state: TrajectoryCollectionState[_EnvState],
        step_index: jax.Array,
        actor_params: FrozenVariableDict,
        critic_params: FrozenVariableDict,
    ):
        # Get keys for sampling action and stepping environment
        # doing it this way to try to get *exactly* the same rngs as in rejax.PPO.
        rng, new_rngs = jax.random.split(collection_state.rng, 2)
        rng_steps, rng_action = jax.random.split(new_rngs, 2)
        rng_steps = jax.random.split(rng_steps, self.hp.num_envs)

        # Sample action
        unclipped_action, log_prob = self.actor.apply(
            actor_params, collection_state.last_obs, rng_action, method="action_log_prob"
        )
        assert isinstance(log_prob, jax.Array)
        value = self.critic.apply(critic_params, collection_state.last_obs)
        assert isinstance(value, jax.Array)

        # Clip action
        if self.discrete:
            action = unclipped_action
        else:
            low = self.env.action_space(self.env_params).low
            high = self.env.action_space(self.env_params).high
            action = jnp.clip(unclipped_action, low, high)

        # Step environment
        next_obs, env_state, reward, done, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            rng_steps,
            collection_state.env_state,
            action,
            self.env_params,
        )

        if self.hp.normalize_observations:
            # rms_state, next_obs = learner.update_and_normalize(collection_state.rms_state, next_obs)
            rms_state = _update_rms(collection_state.rms_state, obs=next_obs, batched=True)
            next_obs = _normalize_obs(rms_state, obs=next_obs)

            collection_state = collection_state.replace(rms_state=rms_state)

        # Return updated runner state and transition
        transition = Trajectory(
            collection_state.last_obs, unclipped_action, log_prob, reward, value, done
        )
        collection_state = collection_state.replace(
            env_state=env_state,
            last_obs=next_obs,
            last_done=done,
            global_step=collection_state.global_step + self.hp.num_envs,
            rng=rng,
        )
        return collection_state, transition

    @property
    def discrete(self) -> bool:
        return isinstance(
            self.env.action_space(self.env_params), gymnax.environments.spaces.Discrete
        )

    def visualize(self, ts: PPOState, gif_path: str | Path):
        actor = make_actor(ts=ts, hp=self.hp)
        render_episode(
            actor=actor, env=self.env, env_params=self.env_params, gif_path=Path(gif_path)
        )


def has_discrete_actions(
    env: Environment[gymnax.EnvState, _EnvParams], env_params: _EnvParams
) -> bool:
    return isinstance(env.action_space(env_params), gymnax.environments.spaces.Discrete)


def _update_rms(rms_state: RMSState, obs: jax.Array, batched: bool = True):
    batch = obs if batched else jnp.expand_dims(obs, 0)

    batch_count = batch.shape[0]
    batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)

    delta = batch_mean - rms_state.mean
    tot_count = rms_state.count + batch_count

    new_mean = rms_state.mean + delta * batch_count / tot_count
    m_a = rms_state.var * rms_state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * rms_state.count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return RMSState(mean=new_mean, var=new_var, count=new_count)


def _normalize_obs(rms_state: RMSState, obs: jax.Array):
    return (obs - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)


@functools.partial(jit, static_argnames=["num_minibatches"])
def shuffle_and_split(
    data: AdvantageMinibatch, rng: chex.PRNGKey, num_minibatches: int
) -> AdvantageMinibatch:
    assert data.trajectories.obs.shape
    iteration_size = data.trajectories.obs.shape[0] * data.trajectories.obs.shape[1]
    permutation = jax.random.permutation(rng, iteration_size)
    _shuffle_and_split_fn = functools.partial(
        _shuffle_and_split,
        permutation=permutation,
        num_minibatches=num_minibatches,
    )
    return jax.tree.map(_shuffle_and_split_fn, data)


@functools.partial(jit, static_argnames=["num_minibatches"])
def _shuffle_and_split(x: jax.Array, permutation: jax.Array, num_minibatches: Static[int]):
    x = x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))
    x = jnp.take(x, permutation, axis=0)
    return x.reshape(num_minibatches, -1, *x.shape[1:])


@jit
def calculate_gae(
    trajectories: TrajectoryWithLastObs,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    get_advantages_fn = functools.partial(get_advantages, gamma=gamma, gae_lambda=gae_lambda)
    _, advantages = jax.lax.scan(
        get_advantages_fn,
        init=(jnp.zeros_like(last_val), last_val),
        xs=trajectories,
        reverse=True,
    )
    return advantages, advantages + trajectories.trajectories.value


@jit
def get_advantages(
    advantage_and_next_value: tuple[jax.Array, jax.Array],
    transition: TrajectoryWithLastObs,
    gamma: float,
    gae_lambda: float,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    advantage, next_value = advantage_and_next_value
    transition_data = transition.trajectories
    assert isinstance(transition_data.reward, jax.Array)
    delta = (
        transition_data.reward.squeeze()  # For gymnax envs that return shape (1, )
        + gamma * next_value * (1 - transition_data.done)
        - transition_data.value
    )
    advantage = delta + gamma * gae_lambda * (1 - transition_data.done) * advantage
    assert isinstance(transition_data.value, jax.Array)
    return (advantage, transition_data.value), advantage


@functools.partial(jit, static_argnames=["actor"])
def actor_loss_fn(
    params: FrozenVariableDict,
    actor: Static[flax.linen.Module],
    batch: AdvantageMinibatch,
    clip_eps: float,
    ent_coef: float,
) -> jax.Array:
    log_prob, entropy = actor.apply(
        params,
        batch.trajectories.obs,
        batch.trajectories.action,
        method="log_prob_entropy",
    )
    assert isinstance(entropy, jax.Array)
    entropy = entropy.mean()

    # Calculate actor loss
    ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pi_loss1 = ratio * advantages
    pi_loss2 = clipped_ratio * advantages
    pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
    return pi_loss - ent_coef * entropy


@functools.partial(jit, static_argnames=["critic"])
def critic_loss_fn(
    params: FrozenVariableDict,
    critic: Static[flax.linen.Module],
    batch: AdvantageMinibatch,
    clip_eps: float,
    vf_coef: float,
):
    value = critic.apply(params, batch.trajectories.obs)
    assert isinstance(value, jax.Array)
    value_pred_clipped = batch.trajectories.value + (value - batch.trajectories.value).clip(
        -clip_eps, clip_eps
    )
    assert isinstance(value_pred_clipped, jax.Array)
    value_losses = jnp.square(value - batch.targets)
    value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    return vf_coef * value_loss


def _actor(
    obs: jax.Array,
    rng: chex.PRNGKey,
    actor_ts: TrainState,
    rms_state: RMSState | None,
    normalize_observations: bool,
):
    if normalize_observations:
        assert rms_state is not None
        obs = _normalize_obs(rms_state, obs)

    obs = jnp.expand_dims(obs, 0)
    action = actor_ts.apply_fn(actor_ts.params, obs, rng, method="act")
    return jnp.squeeze(action)


def make_actor(
    ts: PPOState[Any], hp: PPOHParams
) -> Callable[[jax.Array, chex.PRNGKey], jax.Array]:
    return functools.partial(
        _actor,
        actor_ts=ts.actor_ts,
        rms_state=ts.data_collection_state.rms_state,
        normalize_observations=hp.normalize_observations,
    )


def render_episode(
    actor: Callable[[jax.Array, chex.PRNGKey], jax.Array],
    env: Environment[Any, _EnvParams],
    env_params: _EnvParams,
    gif_path: Path,
    rng: chex.PRNGKey = jax.random.key(123),
    num_steps: int = 200,
):
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    for step in range(num_steps):
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = actor(obs, rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            key=rng_step, state=env_state, action=action, params=env_params
        )
        reward_seq.append(reward)
        # if done or step >= 500:
        #     break
        obs = next_obs
        env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    # gif_path = Path(log_dir) / f"epoch_{current_epoch}.gif"
    logger.info(f"Saving gif to {gif_path}")
    # print(f"Saving gif to {gif_path}")
    with contextlib.redirect_stderr(None):
        vis.animate(str(gif_path))
    plt.close(vis.fig)


## Pytorch-Lightning wrapper around this learner:


class JaxRlExample(lightning.LightningModule):
    """Example of a RL algorithm written in Jax, in this case, PPO.

    This is an un-folded version of `rejax.PPO`.
    """

    def __init__(
        self,
        learner: PPOLearner,
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
        actor_losses = train_metrics["actor_losses"]
        critic_losses = train_metrics["critic_losses"]
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
        eval_metrics = self.learner.eval_callback(ts=self.train_state)
        episode_lengths = eval_metrics.episode_length
        cumulative_rewards = eval_metrics.cumulative_reward
        self.log("val/episode_lengths", episode_lengths.mean().item(), batch_size=1)
        self.log("val/rewards", cumulative_rewards.mean().item(), batch_size=1)

    def on_train_epoch_start(self) -> None:
        if not isinstance(self.env, gymnax.environments.environment.Environment):
            return
        assert self.trainer.log_dir is not None
        gif_path = Path(self.trainer.log_dir) / f"epoch_{self.current_epoch}.gif"
        self.learner.visualize(ts=self.train_state, gif_path=gif_path)
        return  # skip the rest for now while we compare the performance
        actor = make_actor(ts=self.train_state, hp=self.hp)
        render_episode(
            actor=actor,
            env=self.env,
            env_params=self.env_params,
            gif_path=gif_path,
            num_steps=200,
        )
        return super().on_train_epoch_end()

    @override
    def configure_optimizers(self) -> Any:
        # todo: Note, this one isn't used atm!
        from torch.optim.adam import Adam

        return Adam(self.parameters(), lr=1e-3)

    @override
    def configure_callbacks(self) -> list[lightning.Callback]:
        return [RlThroughputCallback()]

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


class RenderEpisodesCallback(JaxCallback):
    on_every_epoch: int = False

    def on_fit_start(self, trainer: JaxTrainer, module: PPOLearner, ts: PPOState):
        if not self.on_every_epoch:
            return
        log_dir = trainer.logger.save_dir if trainer.logger else trainer.default_root_dir
        assert log_dir is not None
        gif_path = Path(log_dir) / f"step_{ts.data_collection_state.global_step:05}.gif"
        module.visualize(ts=ts, gif_path=gif_path)

    def on_train_epoch_start(self, trainer: JaxTrainer, module: PPOLearner, ts: PPOState):
        if not self.on_every_epoch:
            return
        log_dir = trainer.logger.save_dir if trainer.logger else trainer.default_root_dir
        assert log_dir is not None
        gif_path = Path(log_dir) / f"epoch_{ts.data_collection_state.global_step:05}.gif"
        module.visualize(ts=ts, gif_path=gif_path)


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

    def log(
        self,
        name: str,
        value: Any,
        module: PPOLearner,
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


def main():
    env_id = "Pendulum-v1"
    env_id = gymnax.environments.classic_control.pendulum.Pendulum
    # env_id = "halfcheetah"
    # env_id = "humanoid"

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

    algo = PPOLearner(
        env=env,
        env_params=env_params,
        actor=PPOLearner.create_actor(env, env_params),
        critic=PPOLearner.create_critic(),
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
        ),
    )
    # train_pure_jax(algo, backend="cpu")
    # train_rejax(env=algo.env, env_params=algo.env_params, hp=algo.hp, backend="cpu")
    # train_lightning(algo, accelerator="cpu")
    rng = jax.random.key(123)

    train_pure_jax(algo, rng=rng, backend=None, n_agents=None)
    # train_rejax(env=algo.env, env_params=algo.env_params, hp=algo.hp, backend=None, rng=rng)
    # train_lightning(algo, accelerator="cuda", devices=1)

    return


def train_pure_jax(
    algo: PPOLearner, rng: chex.PRNGKey, n_agents: int | None = None, backend: str | None = None
):
    print("Pure Jax (ours)")
    print("Compiling...")
    start = time.perf_counter()
    import jax._src.deprecations

    jax._src.deprecations._registered_deprecations["tracer-hash"].accelerated = True
    # num_epochs = np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int)
    max_epochs: int = np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int)

    iteration_steps = algo.hp.num_envs * algo.hp.num_steps
    num_iterations = np.ceil(algo.hp.eval_freq / iteration_steps).astype(int)
    training_steps_per_epoch: int = num_iterations

    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
    from lightning.pytorch.loggers import CSVLogger

    lightning.Trainer.log_dir
    trainer = JaxTrainer(
        max_epochs=max_epochs,
        training_steps_per_epoch=training_steps_per_epoch,
        logger=CSVLogger(save_dir="logs/jax_rl_debug", name=None, flush_logs_every_n_steps=1),
        default_root_dir=REPO_ROOTDIR / "logs",
        callbacks=[
            # Can't use callbacks when using `vmap`!
            # RlThroughputCallback(),
            RenderEpisodesCallback(on_every_epoch=False),
            RichProgressBar(),
        ],
    )
    train_fn = functools.partial(trainer.fit)

    if n_agents:
        train_fn = jax.vmap(train_fn)
        rng = jax.random.split(rng, n_agents)

    train_fn = jax.jit(train_fn, backend=backend).lower(algo, rng).compile()
    print(f"Finished compiling in {time.perf_counter() - start} seconds.")

    print("Training" + (f" {n_agents} agents in parallel" if n_agents else "") + "...")
    start = time.perf_counter()
    train_state, evaluations = train_fn(algo, rng)
    jax.block_until_ready((train_state, evaluations))
    print(f"Finished training in {time.perf_counter() - start} seconds.")
    assert isinstance(train_state, PPOState)
    assert isinstance(evaluations, EvalMetrics)

    if n_agents is None:
        algo.visualize(ts=train_state, gif_path=Path("pure_jax.gif"))
    else:
        # Visualize the first agent
        first_agent_ts: PPOState = jax.tree.map(operator.itemgetter(0), train_state)
        algo.visualize(ts=first_agent_ts, gif_path=Path("pure_jax_first.gif"))
        last_evals = evaluations.cumulative_reward[:, -1]
        best_agent = jnp.argmax(last_evals)
        print(f"Best agent is #{best_agent}, with rng: {rng.at[best_agent]}")

        best_agent_ts: PPOState = jax.tree.map(operator.itemgetter(best_agent), train_state)
        algo.visualize(ts=best_agent_ts, gif_path=Path("pure_jax_best.gif"))

        algo.visualize(
            ts=first_agent_ts.replace(
                actor_ts=train_state.actor_ts.replace(
                    params=jax.tree.map(
                        lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else x,
                        train_state.actor_ts.params,
                    )
                ),
                critic_ts=train_state.critic_ts.replace(
                    params=jax.tree.map(
                        lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else x,
                        train_state.critic_ts.params,
                    )
                ),
            ),
            gif_path=Path("pure_jax_avg.gif"),
        )


def train_lightning(
    algo: PPOLearner,
    rng: chex.PRNGKey,
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
):
    # Fit with pytorch-lightning.
    print("Lightning")

    module = JaxRlExample(
        learner=algo,
        ts=algo.init_train_state(rng),
    )

    from project.utils.env_vars import REPO_ROOTDIR

    num_evals = int(np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int))
    trainer = lightning.Trainer(
        max_epochs=num_evals,
        # logger=CSVLogger(save_dir="logs/jax_rl_debug"),
        accelerator=accelerator,
        devices=devices,
        default_root_dir=REPO_ROOTDIR / "logs",
        # reload_dataloaders_every_n_epochs=1,  # todo: use this if we end up making a generator in train_dataloader
        barebones=True,
    )
    start = time.perf_counter()
    trainer.fit(module)
    print(f"Trained in {time.perf_counter() - start:.1f} seconds.")
    train_state = module.train_state
    module.visualize(train_state, gif_path=Path("lightning.gif"))


def train_rejax(
    env: Environment[gymnax.EnvState, _EnvParams],
    env_params: _EnvParams,
    hp: PPOHParams,
    rng: chex.PRNGKey,
    backend: str | None = None,
):
    print("Rejax")
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
    train_fn = jax.jit(algo.train, backend=backend).lower(rng).compile()
    print(f"Compiled in {time.perf_counter() - start} seconds.")
    print("Training...")
    start = time.perf_counter()
    ts, eval = train_fn(rng)
    jax.block_until_ready(ts)
    print(f"Finished training in {time.perf_counter() - start} seconds.")

    actor_ts = ts.actor_ts.replace(apply_fn=algo.actor.apply)
    actor = functools.partial(
        _actor,
        actor_ts=actor_ts,
        rms_state=ts.rms_state,
        normalize_observations=algo.normalize_observations,
    )
    render_episode(actor=actor, env=env, env_params=env_params, gif_path=Path("rejax.gif"))

    # print(ts)


if __name__ == "__main__":
    # train_rejax()
    main()
    exit()

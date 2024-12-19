"""Example of an RL algorithm (PPO) written entirely in Jax.

This is based on [`rejax.PPO`](https://github.com/keraJLi/rejax/blob/main/rejax/algos/ppo.py).
See the `JaxRLExample` class for a description of the differences w.r.t. `rejax.PPO`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import operator
from collections.abc import Callable, Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Generic, TypedDict

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
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment
from gymnax.visualize.visualizer import Visualizer
from matplotlib import pyplot as plt
from rejax.algos.mixins import RMSState
from rejax.evaluate import evaluate
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork
from typing_extensions import TypeVar
from xtils.jitpp import Static

from project import experiment
from project.configs.config import Config
from project.trainers.jax_trainer import JaxCallback, JaxModule, JaxTrainer
from project.utils.typing_utils.jax_typing_utils import field, jit

logger = get_logger(__name__)

TEnvParams = TypeVar("TEnvParams", bound=gymnax.EnvParams, default=gymnax.EnvParams)
"""Type variable for the env params (`gymnax.EnvParams`)."""

TEnvState = TypeVar("TEnvState", bound=gymnax.EnvState, default=gymnax.EnvState)


class Trajectory(flax.struct.PyTreeNode):
    """A sequence of interactions between an agent and an environment."""

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    value: jax.Array
    done: jax.Array


class TrajectoryWithLastObs(flax.struct.PyTreeNode):
    """Trajectory with the last observation and whether the last step is the end of an episode."""

    trajectories: Trajectory
    last_done: jax.Array
    last_obs: jax.Array


class AdvantageMinibatch(flax.struct.PyTreeNode):
    """Annotated trajectories with advantages and targets for the critic."""

    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class TrajectoryCollectionState(Generic[TEnvState], flax.struct.PyTreeNode):
    """Struct containing the state related to the collection of data from the environment."""

    last_obs: jax.Array
    env_state: TEnvState
    rms_state: RMSState
    last_done: jax.Array
    global_step: int
    rng: chex.PRNGKey


class PPOState(Generic[TEnvState], flax.struct.PyTreeNode):
    """Contains all the state of the `JaxRLExample` algorithm."""

    actor_ts: TrainState
    critic_ts: TrainState
    rng: chex.PRNGKey
    data_collection_state: TrajectoryCollectionState[TEnvState]


class PPOHParams(flax.struct.PyTreeNode):
    """Hyper-parameters for this PPO example.

    These are taken from `rejax.PPO` algorithm class.
    """

    num_epochs: int = field(default=8)
    num_envs: int = field(default=64)  # overwrite default
    num_steps: int = field(default=64)
    num_minibatches: int = field(default=16)

    # ADDED:
    num_seeds_per_eval: int = field(default=128)

    eval_freq: int = field(default=4_096)

    normalize_observations: bool = field(default=False)
    total_timesteps: int = field(default=131_072)
    debug: bool = field(default=False)

    learning_rate: chex.Scalar = 0.0003
    gamma: chex.Scalar = 0.99
    max_grad_norm: chex.Scalar = jnp.inf
    # todo: this `jnp.inf` is causing issues in the yaml schema because it becomes `Infinity`.

    gae_lambda: chex.Scalar = 0.95
    clip_eps: chex.Scalar = 0.2
    vf_coef: chex.Scalar = 0.5
    ent_coef: chex.Scalar = 0.01

    # IDEA: Split up the RNGs for different parts?
    # rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(0))
    # networks_rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(1))
    # env_rng: chex.PRNGKey = flax.struct.field(pytree_node=True, default=jax.random.key(2))


class _AgentKwargs(TypedDict):
    activation: str
    hidden_layer_sizes: Sequence[int]


class _NetworkConfig(TypedDict):
    agent_kwargs: _AgentKwargs


class TrainStepMetrics(flax.struct.PyTreeNode):
    actor_losses: jax.Array
    critic_losses: jax.Array


class EvalMetrics(flax.struct.PyTreeNode):
    episode_length: jax.Array
    cumulative_reward: jax.Array


class JaxRLExample(
    flax.struct.PyTreeNode,
    JaxModule[PPOState[TEnvState], TrajectoryWithLastObs, EvalMetrics],
    Generic[TEnvState, TEnvParams],
):
    """Example of an RL algorithm written in Jax: PPO, based on `rejax.PPO`.

    ## Differences w.r.t. rejax.PPO:

    - The state / hparams are split into different, fully-typed structs:
        - The algorithm state is in a typed `PPOState` struct (vs an untyped,
            dynamically-generated struct in rejax).
        - The hyper-parameters are in a typed `PPOHParams` struct.
        - The state variables related to the collection of data from the environment is a
            `TrajectoryCollectionState` instead of everything being bunched up together.
            - This makes it easier to call the `collect_episodes` function with just what it needs.
    - The seeds for the networks and the environment data collection are separated.

    The logic is exactly the same: The losses / updates are computed in the exact same way.
    """

    env: Environment[TEnvState, TEnvParams] = flax.struct.field(pytree_node=False)
    env_params: TEnvParams
    actor: flax.linen.Module = flax.struct.field(pytree_node=False)
    critic: flax.linen.Module = flax.struct.field(pytree_node=False)
    hp: PPOHParams

    @classmethod
    def create(
        cls,
        env_id: str | None = None,
        env: Environment[TEnvState, TEnvParams] | None = None,
        env_params: TEnvParams | None = None,
        hp: PPOHParams | None = None,
    ) -> JaxRLExample[TEnvState, TEnvParams]:
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
            hp=hp or PPOHParams(),
        )

    @classmethod
    def create_networks(
        cls,
        env: Environment[gymnax.EnvState, TEnvParams],
        env_params: TEnvParams,
        config: _NetworkConfig,
    ):
        # Equivalent to:
        # return rejax.PPO.create_agent(config, env, env_params)
        return {
            "actor": cls.create_actor(env, env_params, **config["agent_kwargs"]),
            "critic": cls.create_actor(env, env_params, **config["agent_kwargs"]),
        }

    _TEnvParams = TypeVar("_TEnvParams", bound=gymnax.EnvParams, covariant=True)
    _TEnvState = TypeVar("_TEnvState", bound=gymnax.EnvState, covariant=True)

    @classmethod
    def create_actor(
        cls,
        env: Environment[_TEnvState, _TEnvParams],
        env_params: _TEnvParams,
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

    def init_train_state(self, rng: chex.PRNGKey) -> PPOState[TEnvState]:
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

    # @jit
    def training_step(self, batch_idx: int, ts: PPOState[TEnvState], batch: TrajectoryWithLastObs):
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

    # @jit
    def ppo_update_epoch(
        self, ts: PPOState[TEnvState], epoch_index: int, trajectories: TrajectoryWithLastObs
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

    # @jit
    def ppo_update(self, ts: PPOState[TEnvState], batch: AdvantageMinibatch):
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

    def eval_callback(
        self, ts: PPOState[TEnvState], rng: chex.PRNGKey | None = None
    ) -> EvalMetrics:
        if rng is None:
            rng = ts.rng
        actor = make_actor(ts=ts, hp=self.hp)
        ep_lengths, cum_rewards = evaluate(
            actor,
            ts.rng,
            self.env,
            self.env_params,
            num_seeds=self.hp.num_seeds_per_eval,
            max_steps_in_episode=self.env_params.max_steps_in_episode,
        )
        return EvalMetrics(episode_length=ep_lengths, cumulative_reward=cum_rewards)

    def get_batch(
        self, ts: PPOState[TEnvState], batch_idx: int
    ) -> tuple[PPOState[TEnvState], TrajectoryWithLastObs]:
        data_collection_state, trajectories = self.collect_trajectories(
            ts.data_collection_state,
            actor_params=ts.actor_ts.params,
            critic_params=ts.critic_ts.params,
        )
        ts = ts.replace(data_collection_state=data_collection_state)
        return ts, trajectories

    # @jit
    def collect_trajectories(
        self,
        collection_state: TrajectoryCollectionState[TEnvState],
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

    # @jit
    def env_step(
        self,
        collection_state: TrajectoryCollectionState[TEnvState],
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

    def visualize(self, ts: PPOState, gif_path: str | Path, eval_rng: chex.PRNGKey | None = None):
        actor = make_actor(ts=ts, hp=self.hp)
        render_episode(
            actor=actor,
            env=self.env,
            env_params=self.env_params,
            gif_path=Path(gif_path),
            rng=eval_rng if eval_rng is not None else ts.rng,
        )

    ## These here aren't currently used. They are here to mirror rejax.PPO where the training loop
    # is in the algorithm.

    @functools.partial(jit, static_argnames=["skip_initial_evaluation"])
    def train(
        self,
        rng: jax.Array,
        train_state: PPOState[TEnvState] | None = None,
        skip_initial_evaluation: bool = False,
    ) -> tuple[PPOState[TEnvState], EvalMetrics]:
        """Full training loop in jax.

        This is only here to match the API of `rejax.PPO.train`. This doesn't get called when using
        the `JaxTrainer`, since `JaxTrainer.fit` already does the same thing, but also with support
        for some `JaxCallback`s (as well as some `lightning.Callback`s!).

        Unfolded version of `rejax.PPO.train`.
        """
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state if train_state is not None else self.init_train_state(rng)

        initial_evaluation: EvalMetrics | None = None
        if not skip_initial_evaluation:
            initial_evaluation = self.eval_callback(ts)

        num_evals = np.ceil(self.hp.total_timesteps / self.hp.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(
            self._training_epoch,
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

    # @jit
    def _training_epoch(
        self, ts: PPOState[TEnvState], epoch: int
    ) -> tuple[PPOState[TEnvState], EvalMetrics]:
        # Run a few training iterations
        iteration_steps = self.hp.num_envs * self.hp.num_steps
        num_iterations = np.ceil(self.hp.eval_freq / iteration_steps).astype(int)
        ts = jax.lax.fori_loop(
            0,
            num_iterations,
            # drop metrics for now
            lambda i, train_state_i: self._fused_training_step(i, train_state_i)[0],
            ts,
        )
        # Run evaluation
        return ts, self.eval_callback(ts)

    # @jit
    def _fused_training_step(self, iteration: int, ts: PPOState[TEnvState]):
        """Fused training step in jax (joined data collection + training).

        This is the equivalent of the training step from rejax.PPO. It is only used in tests to
        verify the correctness of the training step.
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


def has_discrete_actions(
    env: Environment[gymnax.EnvState, TEnvParams], env_params: TEnvParams
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


# @jit
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


# @jit
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
    env: Environment[Any, TEnvParams],
    env_params: TEnvParams,
    gif_path: Path,
    rng: chex.PRNGKey,
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
    # Disable the "ffmpeg moviewriter not available, using Pillow" print to stderr that happens in
    # there.
    gif_path.parent.mkdir(exist_ok=True, parents=True)
    with contextlib.redirect_stderr(None):
        vis.animate(str(gif_path))
    plt.close(vis.fig)


class RenderEpisodesCallback(JaxCallback):
    on_every_epoch: bool = False

    def on_fit_start(self, trainer: JaxTrainer, module: JaxRLExample, ts: PPOState):  # type: ignore
        if not self.on_every_epoch:
            return
        log_dir = trainer.logger.save_dir if trainer.logger else trainer.default_root_dir
        assert log_dir is not None
        gif_path = Path(log_dir) / f"step_{ts.data_collection_state.global_step:05}.gif"
        module.visualize(ts=ts, gif_path=gif_path)
        jax.debug.print("Saved gif to {gif_path}", gif_path=gif_path)

    def on_train_epoch_start(self, trainer: JaxTrainer, module: JaxRLExample, ts: PPOState):  # type: ignore
        if not self.on_every_epoch:
            return
        log_dir = trainer.logger.save_dir if trainer.logger else trainer.default_root_dir
        assert log_dir is not None
        gif_path = Path(log_dir) / f"epoch_{ts.data_collection_state.global_step:05}.gif"
        module.visualize(ts=ts, gif_path=gif_path)
        jax.debug.print("Saved gif to {gif_path}", gif_path=gif_path)


@experiment.evaluate.register
def evaluate_ppo_example(
    algorithm: JaxRLExample,
    /,
    *,
    trainer: JaxTrainer,
    train_results: tuple[PPOState, EvalMetrics],
    config: Config,
    datamodule: None = None,
):
    """Override for the `evaluate` function used by `main.py`, in the case of this algorithm."""
    # todo: there isn't yet a `validate` method on the jax trainer.
    assert isinstance(algorithm, JaxModule)
    assert isinstance(trainer, JaxTrainer)
    assert train_results is not None
    metrics = train_results[1]

    last_epoch_metrics = jax.tree.map(operator.itemgetter(-1), metrics)
    assert isinstance(last_epoch_metrics, EvalMetrics)
    # Average across eval seeds (we're doing evaluation in multiple environments in parallel with
    # vmap).
    last_epoch_average_cumulative_reward = last_epoch_metrics.cumulative_reward.mean().item()
    return (
        "-avg_cumulative_reward",
        -last_epoch_average_cumulative_reward,  # need to return an "error" to minimize for HPO.
        dataclasses.asdict(last_epoch_metrics),
    )

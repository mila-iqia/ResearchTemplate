import contextlib
import dataclasses
import functools
import time
from collections.abc import Callable, Iterable, Sequence
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, ParamSpec

import chex
import flax.core
import flax.linen
import flax.struct
import gymnax
import gymnax.environments.spaces
import gymnax.experimental.rollout
import jax
import jax.numpy as jnp
import lightning
import numpy as np
import optax
import rejax
import rejax.evaluate
import torch
import torch_jax_interop
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from jax._src.sharding_impls import UNSPECIFIED, Device
from rejax.algos.mixins import RMSState
from torch.utils.data import DataLoader, IterableDataset
from typing_extensions import Self, TypeVar, override
from xtils import jitpp
from xtils.jitpp import Static

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.algorithms.jax_example import JaxFcNet

logger = get_logger(__name__)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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


class TrajectoryCollectionState(flax.struct.PyTreeNode):
    last_obs: jax.Array
    env_state: gymnax.EnvState
    rms_state: RMSState
    last_done: jax.Array
    global_step: int
    rng: chex.PRNGKey


class PPOState(flax.struct.PyTreeNode):
    actor_ts: TrainState
    critic_ts: TrainState
    rng: chex.PRNGKey
    data_collection_state: TrajectoryCollectionState


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


class JaxRlExample(lightning.LightningModule):
    """Example of a RL algorithm written in Jax, in this case, PPO.

    This is an un-folded version of `rejax.PPO`.
    """

    class HParams(flax.struct.PyTreeNode):
        # TODO: Need to rename a few of these to make it less confusing.
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

    def __init__(
        self,
        env: gymnax.environments.environment.Environment[TEnvState, TEnvParams],
        env_params: TEnvParams,
        hp: HParams | None = None,
    ):
        # https://github.com/keraJLi/rejax/blob/a1428ad3d661e31985c5c19460cec70bc95aef6e/configs/gymnax/pendulum.yaml#L1

        super().__init__()
        self.env = env
        self.env_params = env_params
        self.hp = hp or self.HParams()
        self.save_hyperparameters(
            {"hp": dataclasses.asdict(self.hp)}, ignore=["env", "env_params"]
        )

        sample_obs = env.observation_space(self.env_params).sample(jax.random.key(1))
        action_shape = env.action_space(self.env_params).shape

        net_rng = jax.random.key(123)
        self.network = JaxFcNet(num_classes=int(np.prod(action_shape)), num_features=256)
        self.network_params = self.network.init(net_rng, sample_obs, forward_rng=None)
        self.rollout_rng_key = jax.random.key(123)

        self.torch_network_params = torch.nn.ParameterList(
            jax.tree.leaves(
                jax.tree.map(torch_jax_interop.to_torch.jax_to_torch_tensor, self.network_params)
            )
        )

        self.automatic_optimization = False

        _agents = rejax.PPO.create_agent(config={}, env=self.env, env_params=self.env_params)
        self.actor: flax.linen.Module = _agents["actor"]
        self.critic: flax.linen.Module = _agents["critic"]

        iteration_steps = self.hp.num_envs * self.hp.num_steps
        # number of "iterations" (collecting batches of episodes in the environment) per epoch.
        self.num_train_iterations = np.ceil(self.hp.eval_freq / iteration_steps).astype(int)
        # todo: number of epochs:
        # num_evals = np.ceil(self.learner.total_timesteps / self.learner.eval_freq).astype(int)

        self.train_state = init_train_state(
            env=env,
            env_params=env_params,
            actor=self.actor,
            critic=self.critic,
            hp=self.hp,
            networks_rng=jax.random.key(123),
            env_rng=jax.random.key(0),
        )

    @override
    def train_dataloader(self) -> Iterable[Trajectory]:
        dataset = EnvDataset(
            env=self.env,
            env_params=self.env_params,
            actor=self.actor,
            critic=self.critic,
            # current_epoch=self.train_state.step # TODO: Use something in the train state.
            current_epoch=self.current_epoch,
            hp=self.hp,
            train_state=self.train_state,
            num_train_iterations=self.num_train_iterations,
        )

        return DataLoader(dataset, num_workers=0, batch_size=None, shuffle=False, collate_fn=None)

    @override
    def training_step(self, batch: TrajectoryWithLastObs, batch_idx: int):
        shapes = jax.tree.map(jnp.shape, batch)
        logger.debug(f"Shapes: {shapes}")

        ts = self.train_state

        trajectories = batch

        # Perhaps instead of doing it this way, we could just get the losses, the grads, then put
        # them on the torch params .grad attribute (hopefully the .data is pointing to the jax
        # tensors, not a copy, so we dont use extra memory). This would perhaps make this
        # compatible with pytorch-lightning manual optimization.

        # Note: This scan is equivalent to a for loop (8 "epochs")
        # while the other scan in `ppo_update_epoch` is a for loop over minibatches.
        start = time.perf_counter()
        with jax.disable_jit(self.hp.debug):
            ts, (actor_losses, critic_losses) = jax.lax.scan(
                functools.partial(self.ppo_update_epoch, trajectories=trajectories),
                init=ts,
                xs=jnp.arange(self.hp.num_epochs),  # type: ignore
                length=self.hp.num_epochs,
            )
        duration = time.perf_counter() - start
        updates_per_second = (self.hp.num_epochs * self.hp.num_minibatches) / duration
        self.log("train/updates_per_second", updates_per_second, logger=True, prog_bar=True)
        minibatch_size = (self.hp.num_envs * self.hp.num_steps) // self.hp.num_minibatches
        samples_per_update = minibatch_size
        self.log(
            "train/samples_per_second",
            updates_per_second * samples_per_update,
            logger=True,
            prog_bar=True,
            on_step=True,
        )

        self.train_state = ts

        self.log("train/actor_loss", actor_losses.mean().item(), logger=True, prog_bar=True)
        self.log("train/critic_loss", critic_losses.mean().item(), logger=True, prog_bar=True)

    @functools.partial(jit, static_argnames=["self", "iteration"])
    def training_step_jax(self, ts: PPOState, iteration: int):
        """Training step in pure jax (joined data collection + training).

        *MUCH* faster than using pytorch-lightning, but you lose the callbacks and such.
        """
        data_collection_state, trajectories = collect_trajectories(
            env=self.env,
            env_params=self.env_params,
            collection_state=ts.data_collection_state,
            actor=self.actor,
            actor_params=ts.actor_ts.params,
            critic=self.critic,
            critic_params=ts.critic_ts.params,
            num_envs=self.hp.num_envs,
            num_steps=self.hp.num_steps,
            discrete=False,
            normalize_observations=self.hp.normalize_observations,
        )
        ts = ts.replace(data_collection_state=data_collection_state)

        # batch = AdvantageMinibatch(trajectories.trajectories, advantages, targets)

        ts, (actor_losses, critic_losses) = jax.lax.scan(
            functools.partial(self.ppo_update_epoch, trajectories=trajectories),
            init=ts,
            xs=jnp.arange(self.hp.num_epochs),  # type: ignore
            length=self.hp.num_epochs,
        )

        return ts

    @functools.partial(jit, static_argnames=["self"])
    def ppo_update(self, ts: PPOState, batch: AdvantageMinibatch):
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

    @functools.partial(jit, static_argnames=["self", "epoch_index"])
    def ppo_update_epoch(
        self: Static[Self], ts: PPOState, epoch_index: int, trajectories: TrajectoryWithLastObs
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

    def val_dataloader(self) -> Any:
        # todo: unsure what this should be yielding..
        yield from range(2)

    def validation_step(self, batch: int, batch_index: int):
        # self.learner.eval_callback()
        actor = make_actor(ts=self.train_state, hp=self.hp)
        rng = jax.random.key(batch_index)
        max_steps = self.env_params.max_steps_in_episode
        episode_lengths, cumulative_rewards = rejax.evaluate.evaluate(
            actor,
            rng,
            env=self.env,
            env_params=self.env_params,
            num_seeds=128,
            max_steps_in_episode=max_steps,
        )
        assert isinstance(episode_lengths, jax.Array)
        assert isinstance(cumulative_rewards, jax.Array)
        self.log("val/episode_lengths", episode_lengths.mean().item(), batch_size=1)
        self.log("val/rewards", cumulative_rewards.mean().item(), batch_size=1)

    def on_train_epoch_start(self) -> None:
        if not isinstance(self.env, gymnax.environments.environment.Environment):
            return
        actor = make_actor(ts=self.train_state, hp=self.hp)
        assert self.trainer.log_dir is not None
        gif_path = Path(self.trainer.log_dir) / f"epoch_{self.current_epoch}.gif"
        visualize_gymnax(actor=actor, env=self.env, env_params=self.env_params, gif_path=gif_path)
        return super().on_train_epoch_end()

    @override
    def configure_optimizers(self) -> Any:
        # todo: Note, this one isn't used atm!
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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

    def visualize(self, ts: PPOState | None, gif_path: str | Path):
        ts = ts or self.train_state
        actor = make_actor(ts=ts, hp=self.hp)
        visualize_gymnax(
            actor=actor, env=self.env, env_params=self.env_params, gif_path=Path(gif_path)
        )

    @functools.partial(jax.jit, static_argnames=["self", "skip_initial_evaluation"])
    def fit_pure_jax(
        self,
        rng: jax.Array,
        train_state: PPOState | None = None,
        skip_initial_evaluation: bool = False,
    ) -> tuple[PPOState, jax.Array]:
        """Training loop in pure jax (MUCH faster than using pytorch-lightning)."""
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        rng, networks_rng, env_rng = jax.random.split(rng, 3)

        agent_kwargs = {}  # todo: use `self.hp.agent_kwargs` or similar?
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(flax.linen, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        _agents = rejax.PPO.create_agent(config={}, env=self.env, env_params=self.env_params)
        actor: flax.linen.Module = _agents["actor"]
        critic: flax.linen.Module = _agents["critic"]
        # assert actor == self.actor
        # assert critic == self.critic

        # iteration_steps = self.hp.num_envs * self.hp.num_steps
        # number of "iterations" (collecting batches of episodes in the environment) per epoch.
        # self.num_train_iterations = np.ceil(self.hp.eval_freq / iteration_steps).astype(int)
        # todo: number of epochs:
        # num_evals = np.ceil(self.learner.total_timesteps / self.learner.eval_freq).astype(int)

        ts = train_state or init_train_state(
            env=self.env,
            env_params=self.env_params,
            actor=actor,
            critic=critic,
            hp=self.hp,
            networks_rng=networks_rng,
            env_rng=env_rng,
        )
        from rejax.evaluate import evaluate

        def eval_callback(algo: JaxRlExample, ts: PPOState, rng: chex.PRNGKey):
            actor = make_actor(ts=ts, hp=algo.hp)
            max_steps = algo.env_params.max_steps_in_episode
            return evaluate(actor, rng, algo.env, algo.env_params, 128, max_steps)

        initial_evaluation: jax.Array | None = None
        if not skip_initial_evaluation:
            initial_evaluation = eval_callback(self, ts, ts.rng)

        def eval_iteration(ts: PPOState, unused):
            # Run a few training iterations
            iteration_steps = self.hp.num_envs * self.hp.num_steps
            num_iterations = np.ceil(self.hp.eval_freq / iteration_steps).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda iteration, ts: self.training_step_jax(ts, iteration),
                ts,
            )
            rejax.PPO.train_iteration
            # Run evaluation
            return ts, eval_callback(self, ts, ts.rng)

        num_evals = np.ceil(self.hp.total_timesteps / self.hp.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, num_evals)

        if not skip_initial_evaluation:
            assert initial_evaluation is not None
            evaluation = jax.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation


@jitpp.jit
def env_step(
    collection_state: TrajectoryCollectionState,
    step_index: jax.Array,
    num_envs: Static[int],
    actor: Static[flax.linen.Module],
    actor_params: FrozenVariableDict,
    critic: Static[flax.linen.Module],
    critic_params: FrozenVariableDict,
    env: Static[Environment[gymnax.EnvState, TEnvParams]],
    env_params: TEnvParams,
    discrete: Static[bool],
    normalize_observations: Static[bool],
):
    # Get keys for sampling action and stepping environment
    # doing it this way to try to get *exactly* the same rngs as in rejax.PPO.
    rng, new_rngs = jax.random.split(collection_state.rng, 2)
    rng_steps, rng_action = jax.random.split(new_rngs, 2)
    rng_steps = jax.random.split(rng_steps, num_envs)

    # Sample action
    unclipped_action, log_prob = actor.apply(
        actor_params, collection_state.last_obs, rng_action, method="action_log_prob"
    )
    assert isinstance(log_prob, jax.Array)
    value = critic.apply(critic_params, collection_state.last_obs)
    assert isinstance(value, jax.Array)

    # Clip action
    if discrete:
        action = unclipped_action
    else:
        low = env.action_space(env_params).low
        high = env.action_space(env_params).high
        action = jnp.clip(unclipped_action, low, high)

    # Step environment
    next_obs, env_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        key=rng_steps,
        state=collection_state.env_state,
        action=action,
        params=env_params,
    )

    if normalize_observations:
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
        global_step=collection_state.global_step + num_envs,
        rng=rng,
    )
    return collection_state, transition


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


@jitpp.jit
def collect_trajectories(
    env: Static[Environment[TEnvState, TEnvParams]],
    env_params: TEnvParams,
    collection_state: TrajectoryCollectionState,
    *,
    actor: Static[flax.linen.Module],
    actor_params: FrozenVariableDict,
    critic: Static[flax.linen.Module],
    critic_params: FrozenVariableDict,
    num_envs: Static[int],
    num_steps: Static[int],
    discrete: Static[bool],
    normalize_observations: Static[bool],
):
    env_step_fn = functools.partial(
        env_step,
        num_envs=num_envs,
        actor=actor,
        actor_params=actor_params,
        critic=critic,
        critic_params=critic_params,
        env=env,
        env_params=env_params,
        discrete=discrete,
        normalize_observations=normalize_observations,
    )
    collection_state, trajectories = jax.lax.scan(
        env_step_fn,
        collection_state,
        xs=jnp.arange(num_steps),
        length=num_steps,
    )
    trajectories_with_last = TrajectoryWithLastObs(
        trajectories=trajectories,
        last_done=collection_state.last_done,
        last_obs=collection_state.last_obs,
    )
    return collection_state, trajectories_with_last


class EnvDataset(IterableDataset):
    def __init__(
        self,
        # learner: PPOLearner,
        current_epoch: int,
        env_params: gymnax.EnvParams,
        actor: flax.linen.Module,
        critic: flax.linen.Module,
        env: Environment,
        hp: JaxRlExample.HParams,
        train_state: PPOState,
        num_train_iterations: int,
    ):
        # self.learner = learner
        self.current_epoch = current_epoch
        self.collection_state = train_state.data_collection_state
        self.env_params = env_params
        self.env = env
        self.actor = actor
        self.critic = critic
        self.hp = hp
        self.train_state = train_state
        self.num_train_iterations = num_train_iterations

        _action_space = self.env.action_space(self.env_params)
        self.discrete = isinstance(_action_space, gymnax.environments.spaces.Discrete)

    def __len__(self):
        return self.num_train_iterations

    def __iter__(self):
        # env_rng = jax.random.key(self.current_epoch)
        # obs, env_state = rejax.PPOvmap_reset(
        #     jax.random.split(env_rng, self.learner.num_envs), self.learner.env_params
        # )
        # collection_state = TrajectoryCollectionState(
        #     last_obs=obs,
        #     rms_state=RMSState.create(
        #         shape=(1, *self.env.observation_space(self.env_params).shape)
        #     ),
        #     global_step=self.train_state.global_step,
        #     env_state=env_state,
        #     last_done=jnp.zeros(self.learner.num_envs, dtype=bool),
        # )
        for batch_idx in range(self.num_train_iterations):
            # episode_key = jax.random.fold_in(self.collection_state.rng, batch_idx)
            start = time.perf_counter()
            self.collection_state, trajectories = collect_trajectories(
                env=self.env,
                env_params=self.env_params,
                collection_state=self.collection_state,
                num_envs=self.hp.num_envs,
                num_steps=self.hp.num_steps,
                actor=self.actor,
                actor_params=self.train_state.actor_ts.params,
                critic=self.critic,
                critic_params=self.train_state.critic_ts.params,
                discrete=self.discrete,
                normalize_observations=self.hp.normalize_observations,
            )

            duration = time.perf_counter() - start
            logger.debug(
                f"Took {duration} seconds to collect {self.hp.num_steps} steps in {self.hp.num_envs} envs."
            )
            # last_val = self.learner.critic.apply(
            #     self.train_state.critic_ts.params, collection_state.last_obs
            # )
            # assert isinstance(last_val, jax.Array)
            # last_val = jnp.where(collection_state.last_done, 0, last_val)
            # advantages, targets = self.learner.calculate_gae(trajectories, last_val)
            # batch = AdvantageMinibatch(trajectories, advantages, targets)

            # for _epoch in range(self.learner.num_epochs):
            #     epoch_key = jax.random.fold_in(episode_key, _epoch)
            #     minibatches = self.learner.shuffle_and_split(batch, epoch_key)

            #     for i in range(self.learner.num_minibatches):
            #         minibatch = jax.tree.map(operator.itemgetter(i), minibatches)
            #         yield minibatch

            yield trajectories


def init_train_state(
    env: Environment[TEnvState, TEnvParams],
    env_params: TEnvParams,
    actor: flax.linen.Module,
    critic: flax.linen.Module,
    hp: JaxRlExample.HParams,
    networks_rng: chex.PRNGKey,
    env_rng: chex.PRNGKey,
) -> PPOState:
    rng, rng_actor, rng_critic = jax.random.split(networks_rng, 3)
    obs_ph = jnp.empty([1, *env.observation_space(env_params).shape])

    actor_params = actor.init(rng_actor, obs_ph, rng_actor)
    critic_params = critic.init(rng_critic, obs_ph)

    tx = optax.adam(learning_rate=hp.learning_rate)
    # TODO: Why isn't the `apply_fn` not set in rejax?
    actor_ts = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx)
    critic_ts = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(env_rng, hp.num_envs), env_params
    )

    collection_state = TrajectoryCollectionState(
        last_obs=obs,
        rms_state=RMSState.create(shape=(1, *env.observation_space(env_params).shape)),
        global_step=0,
        env_state=env_state,
        last_done=jnp.zeros(hp.num_envs, dtype=bool),
        rng=env_rng,
    )

    return PPOState(
        actor_ts=actor_ts,
        critic_ts=critic_ts,
        rng=rng,
        data_collection_state=collection_state,
    )


@jitpp.jit
def shuffle_and_split(
    data: AdvantageMinibatch, rng: chex.PRNGKey, num_minibatches: Static[int]
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


@jitpp.jit
def _shuffle_and_split(x: jax.Array, permutation: jax.Array, num_minibatches: Static[int]):
    x = x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))
    x = jnp.take(x, permutation, axis=0)
    return x.reshape(num_minibatches, -1, *x.shape[1:])


@jitpp.jit
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


@jitpp.jit
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


@jitpp.jit
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


@jitpp.jit
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


def make_actor(ts: PPOState, hp: JaxRlExample.HParams):
    return functools.partial(
        _actor,
        actor_ts=ts.actor_ts,
        rms_state=ts.data_collection_state.rms_state,
        normalize_observations=hp.normalize_observations,
    )


class RlThroughputCallback(MeasureSamplesPerSecondCallback):
    """A callback to measure the throughput of RL algorithms."""

    def __init__(self):
        super().__init__()
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


def visualize_gymnax(
    actor: Callable[[jax.Array, chex.PRNGKey], jax.Array],
    env: Environment[Any, TEnvParams],
    env_params: TEnvParams,
    gif_path: Path,
    rng: chex.PRNGKey = jax.random.key(123),
    num_steps: int = 100,
):
    from gymnax.visualize.visualizer import Visualizer

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
    with contextlib.redirect_stdout(None):
        vis.animate(str(gif_path))


def main():
    env_id = "Pendulum-v1"
    # env_id = "halfcheetah"
    # env_id = "humanoid"

    from brax.envs import _envs as brax_envs
    from rejax.compat.brax2gymnax import create_brax

    if env_id in brax_envs:
        env, env_params = create_brax(
            env_id,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
            batch_size=None,
            backend="generalized",
        )
    else:
        env, env_params = gymnax.make(env_id=env_id)

    algo = JaxRlExample(
        env=env,
        env_params=env_params,
        hp=JaxRlExample.HParams(
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
    train_fn(algo)

    return


def train_fn(algo: JaxRlExample):
    # num_evals = int(np.ceil(algo.hp.total_timesteps / algo.hp.eval_freq).astype(int))

    print("Compiling...")
    start = time.perf_counter()
    rng = jax.random.key(123)
    train_fn = jax.jit(algo.fit_pure_jax).lower(rng).compile()
    print(f"Finished compiling in {time.perf_counter() - start} seconds.")

    print("Training...")
    start = time.perf_counter()
    train_state, evaluations = train_fn(rng)
    print(f"Finished training in {time.perf_counter() - start} seconds.")
    assert isinstance(train_state, PPOState)
    algo.visualize(train_state, gif_path=Path("pure_jax.gif"))

    # Fit with pytorch-lightning.
    # trainer = lightning.Trainer(
    #     max_epochs=num_evals,
    #     logger=CSVLogger(save_dir="logs/jax_rl_debug"),
    #     devices=1,
    #     default_root_dir=REPO_ROOTDIR / "logs",
    # )
    # # algo.fit(rng)
    # actor = functools.partial(
    #     _actor,
    #     actor_ts=train_state.actor_ts,
    #     rms_state=train_state.darms_state,
    #     normalize_observations=algo.hp.normalize_observations,
    # )

    # visualize_gymnax(actor=actor, env=env, env_params=env_params, gif_path=Path("rejax.gif"))


def train_rejax():
    env, env_params = gymnax.make(env_id="Pendulum-v1")
    algo = rejax.PPO.create(
        env=env,
        env_params=env_params,
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
    )
    print("Compiling...")
    start = time.perf_counter()
    rng = jax.random.key(123)
    train_fn = jax.jit(algo.train).lower(rng).compile()
    print(f"Compiled in {time.perf_counter() - start} seconds.")
    print("Training...")
    start = time.perf_counter()
    ts, eval = train_fn(rng)
    print(f"Finished training in {time.perf_counter() - start} seconds.")

    actor_ts = ts.actor_ts.replace(apply_fn=algo.actor.apply)
    actor = functools.partial(
        _actor,
        actor_ts=actor_ts,
        rms_state=ts.rms_state,
        normalize_observations=algo.normalize_observations,
    )

    rejax.PPO
    visualize_gymnax(actor=actor, env=env, env_params=env_params, gif_path=Path("rejax.gif"))

    # print(ts)


if __name__ == "__main__":
    # train_rejax()
    main()
    exit()

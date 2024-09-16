from __future__ import annotations

import functools
import time
import typing
from collections.abc import Callable, Iterable
from logging import getLogger as get_logger
from typing import Any, Generic, NamedTuple, ParamSpec, Protocol

import chex
import flax.core
import flax.linen
import flax.struct
import gymnax
import gymnax.experimental.rollout
import jax
import jax.numpy as jnp
import lightning
import numpy as np
import rejax
import rejax.evaluate
import torch
import torch_jax_interop
from flax.training.train_state import TrainState
from flax.typing import FrozenVariableDict
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from lightning import LightningModule, Trainer
from rejax.algos.mixins import RMSState
from rejax.algos.ppo import AdvantageMinibatch, Trajectory
from typing_extensions import TypeVar, override

from project.algorithms.callbacks.samples_per_second import MeasureSamplesPerSecondCallback
from project.algorithms.jax_example import JaxFcNet

logger = get_logger(__name__)


P = ParamSpec("P")
Out = TypeVar("Out", covariant=True)


class _Module(Protocol[P, Out]):
    if typing.TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> Out: ...

        init = flax.linen.Module.init
        apply = flax.linen.Module.apply


_ContentType = TypeVar("_ContentType", default=jax.Array)


class _EpisodeStepData(flax.struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


class EpisodeData(flax.struct.PyTreeNode, Generic[_ContentType]):
    obs: _ContentType
    action: _ContentType
    reward: _ContentType
    next_obs: _ContentType
    done: _ContentType
    cum_return: _ContentType


class _StateInput(NamedTuple):
    obs: jax.Array
    state: gymnax.EnvState
    policy_params: flax.core.FrozenDict[str, jax.Array]
    rng: jax.Array
    cum_reward: jax.Array
    valid_mask: jax.Array


class RolloutWrapper(gymnax.experimental.rollout.RolloutWrapper):
    def __init__(
        self,
        env: Environment,
        env_params: gymnax.EnvParams,
        model_forward: Callable,
    ):
        # Define the RL environment & network forward function
        self.env = env
        self.env_params = env_params
        self.model_forward = model_forward

        self.num_env_steps = self.env_params.max_steps_in_episode

    @functools.partial(jax.jit, static_argnums=(0,))
    @override
    def single_rollout(
        self, rng_input: chex.PRNGKey, policy_params: flax.core.FrozenDict[str, jax.Array]
    ):
        """Rollout an episode with lax.scan.

        NOTE: Compared to gymnax.experimental.rollout.RolloutWrapper.single_rollout, this also
        supports other envs than Pendulum.
        """
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(
            state_input: _StateInput, _unused: tuple
        ) -> tuple[_StateInput, _EpisodeStepData]:
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, rng_net)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = _StateInput(
                obs=next_obs,
                state=next_state,
                policy_params=policy_params,
                rng=rng,
                cum_reward=new_cum_reward,
                valid_mask=new_valid_mask,
            )
            out = _EpisodeStepData(obs, action, reward, next_obs, done)
            return carry, out

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            _StateInput(
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ),
            (),
            length=self.env_params.max_steps_in_episode,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        cum_return = carry_out.cum_reward
        return EpisodeData(
            obs=scan_out.obs,
            action=scan_out.action,
            reward=scan_out.reward,
            next_obs=scan_out.next_obs,
            done=scan_out.done,
            cum_return=cum_return,
        )


def get_episodes(
    env: gymnax.environments.environment.Environment[TEnvState, TEnvParams],
    env_params: TEnvParams,
    num_episodes: int,
    model: _Module[[jax.Array, chex.PRNGKey], jax.Array],
    model_params: FrozenVariableDict | dict[str, Any],
    rollout_rng_key: chex.PRNGKey,
) -> EpisodeData:
    rollout_wrapper = RolloutWrapper(
        env=env,
        env_params=env_params,
        model_forward=model.apply,
    )

    rollout_rng_keys = jax.random.split(rollout_rng_key, num_episodes)
    return rollout_wrapper.batch_rollout(rollout_rng_keys, model_params)


class JaxRlExample(lightning.LightningModule):
    def __init__(
        self, env: gymnax.environments.environment.Environment, env_params: gymnax.EnvParams
    ):
        super().__init__()
        self.env = env
        self.env_params = env_params

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

        _agents = PPOLearner.create_agent(config={}, env=self.env, env_params=self.env_params)
        actor = _agents["actor"]
        critic = _agents["critic"]

        # TODO: Super ugly. Remove this.
        def _eval_callback(algo: PPOLearner, ts: PPOTrainState, rng: chex.PRNGKey):
            act = algo.make_act(ts)
            max_steps = algo.env_params.max_steps_in_episode
            return rejax.evaluate.evaluate(
                act,
                rng,
                env=self.env,
                env_params=self.env_params,
                num_seeds=128,
                max_steps_in_episode=max_steps,
            )

        self.learner = PPOLearner(
            env=self.env,
            env_params=self.env_params,
            eval_callback=_eval_callback,
            actor=actor,
            critic=critic,
            # https://github.com/keraJLi/rejax/blob/a1428ad3d661e31985c5c19460cec70bc95aef6e/configs/gymnax/pendulum.yaml#L1
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
        iteration_steps = self.learner.num_envs * self.learner.num_steps
        self.num_train_iterations = np.ceil(self.learner.eval_freq / iteration_steps).astype(int)
        # todo: number of epochs:
        # num_evals = np.ceil(self.learner.total_timesteps / self.learner.eval_freq).astype(int)

        self.train_state = self.learner.init_state(jax.random.key(0))

    @override
    def train_dataloader(self) -> Iterable[Trajectory]:
        env_rng = jax.random.key(self.current_epoch)
        obs, env_state = self.learner.vmap_reset(
            jax.random.split(env_rng, self.learner.num_envs), self.learner.env_params
        )
        collection_state = TrajectoryCollectionState(
            last_obs=obs,
            rms_state=RMSState.create(
                shape=(1, *self.env.observation_space(self.env_params).shape)
            ),
            global_step=self.train_state.global_step,
            env_state=env_state,
            last_done=jnp.zeros(self.learner.num_envs, dtype=bool),
        )
        for batch_idx in range(self.num_train_iterations):
            episode_key = jax.random.fold_in(self.rollout_rng_key, batch_idx)
            start = time.perf_counter()
            collection_state, trajectories = self.learner.collect_trajectories_custom(
                collection_state,
                rng=episode_key,
                actor_params=self.train_state.actor_ts.params,
                critic_params=self.train_state.critic_ts.params,
            )
            duration = time.perf_counter() - start
            print(
                f"Took {duration} seconds to collect {self.learner.num_steps} steps in {self.learner.num_envs} envs."
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
            # yield get_episodes(
            #     env=self.env,
            #     env_params=self.env_params,
            #     num_episodes=10,
            #     model=self.network,
            #     model_params=self.network_params,
            #     rollout_rng_key=episode_key,
            # )

    @override
    def training_step(self, batch: Trajectory, batch_idx: int):
        shapes = jax.tree.map(jnp.shape, batch)
        logger.debug(f"Shapes: {shapes}")

        # initial_actor_state = copy.deepcopy(self.train_state.actor_ts.params)
        # initial_critic_state = copy.deepcopy(self.train_state.critic_ts.params)
        ts = self.train_state

        trajectories = batch

        last_val = self.learner.critic.apply(ts.critic_ts.params, ts.last_obs)
        assert isinstance(last_val, jax.Array)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.learner.calculate_gae(trajectories, last_val)
        batch_with_advantage = AdvantageMinibatch(trajectories, advantages, targets)

        # Perhaps instead of doing it this way, we could just get the losses, the grads, then put
        # them on the torch params .grad attribute (hopefully the .data is pointing to the jax
        # tensors, not a copy, so we dont use extra memory). This would perhaps make this
        # compatible with pytorch-lightning manual optimization.

        # Note: This scan is equivalent to a for loop (8 "epochs")
        # while the other scan in `ppo_update_epoch` is a for loop over minibatches.
        start = time.perf_counter()
        ts, (actor_losses, critic_losses) = jax.lax.scan(
            functools.partial(self.ppo_update_epoch, batch=batch_with_advantage),
            init=ts,
            xs=jnp.arange(self.learner.num_epochs),
            length=self.learner.num_epochs,
        )
        duration = time.perf_counter() - start
        updates_per_second = (self.learner.num_epochs * self.learner.num_minibatches) / duration
        self.log("train/updates_per_second", updates_per_second, logger=True, prog_bar=True)
        samples_per_update = self.learner.minibatch_size
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

    @functools.partial(jax.jit, static_argnames="self")
    def ppo_update(self, ts: PPOTrainState, batch: AdvantageMinibatch):
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
            ts.actor_ts.params,
            actor=self.learner.actor,
            batch=batch,
            clip_eps=self.learner.clip_eps,
            ent_coef=self.learner.ent_coef,
        )
        assert isinstance(actor_loss, jax.Array)
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            ts.critic_ts.params,
            critic=self.learner.critic,
            batch=batch,
            clip_eps=self.learner.clip_eps,
            vf_coef=self.learner.vf_coef,
        )
        assert isinstance(critic_loss, jax.Array)

        # TODO: to log the loss here?
        actor_ts = ts.actor_ts.apply_gradients(grads=actor_grads)
        critic_ts = ts.critic_ts.apply_gradients(grads=critic_grads)

        return ts.replace(actor_ts=actor_ts, critic_ts=critic_ts), (actor_loss, critic_loss)

    @functools.partial(jax.jit, static_argnames="self")
    def ppo_update_epoch(self, ts: PPOTrainState, epoch_index: int, batch: AdvantageMinibatch):
        # shuffle the data and split it into minibatches
        minibatch_rng = jax.random.fold_in(ts.rng, epoch_index)
        minibatches = self.learner.shuffle_and_split(batch, minibatch_rng)
        return jax.lax.scan(self.ppo_update, ts, minibatches, length=self.learner.num_minibatches)

    def val_dataloader(self) -> Any:
        # todo: unsure what this should be yielding..
        yield from range(10)

    def validation_step(self, batch, batch_index: int):
        # self.learner.eval_callback()
        act = self.learner.make_act(self.train_state)
        rng = jax.random.key(batch_index)
        max_steps = self.learner.env_params.max_steps_in_episode
        episode_lengths, cumulative_rewards = rejax.evaluate.evaluate(
            act,
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

    @override
    def configure_optimizers(self) -> Any:
        # todo: Note, this one isn't used atm!
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @override
    def configure_callbacks(self) -> list[MeasureSamplesPerSecondCallback]:
        return [RlThroughputCallback()]

    @override
    def transfer_batch_to_device(
        self, batch: Trajectory, device: torch.device, dataloader_idx: int
    ) -> Trajectory:
        if not isinstance(batch, Trajectory | EpisodeData):
            return batch
        _batch_jax_devices = batch.obs.devices()
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


class PPOTrainState(flax.struct.PyTreeNode):
    actor_ts: TrainState
    critic_ts: TrainState
    rng: chex.PRNGKey
    last_obs: jax.Array
    global_step: int
    env_state: gymnax.EnvState
    last_done: jax.Array
    rms_state: RMSState


# TODO: Use this as an input type to `collect_transitions`, and return the modified obj.
class TrajectoryCollectionState(flax.struct.PyTreeNode):
    last_obs: jax.Array
    env_state: gymnax.EnvState
    rms_state: RMSState
    last_done: jax.Array
    global_step: int


def get_actor_loss_fn(actor: flax.linen.Module):
    @jax.jit
    def actor_loss_fn(
        params: FrozenVariableDict,
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

    return actor_loss_fn


@functools.partial(jax.jit, static_argnames="actor")
def actor_loss_fn(
    params: FrozenVariableDict,
    actor: flax.linen.Module,
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


@functools.partial(jax.jit, static_argnames="critic")
def critic_loss_fn(
    params: FrozenVariableDict,
    critic: flax.linen.Module,
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


class PPOLearner(rejax.PPO):
    """Subclass that just adds some type hints to rejax.PPO."""

    def init_state(self, rng: jax.Array) -> PPOTrainState:
        return super().init_state(rng)

    def make_act(self, ts: PPOTrainState):
        def act(obs: jax.Array, rng: chex.PRNGKey):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    def train_iteration(self, ts):
        return super().train_iteration(ts)

    def calculate_gae(
        self, trajectories: Trajectory, last_val: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return super().calculate_gae(trajectories, last_val)

    def shuffle_and_split(self, data: AdvantageMinibatch, rng: chex.PRNGKey):
        shuffle_split_data = super().shuffle_and_split(data, rng)
        assert isinstance(shuffle_split_data, type(data))
        return shuffle_split_data

    # TODO: Ideally we wouldn't need to return the full PPOTrainState, since it gives the
    # impression that we're changing the weights or something like that. Instead we could/should
    # just return what changed.
    def collect_trajectories_custom(
        self,
        collection_state: TrajectoryCollectionState,
        rng: chex.PRNGKey,
        actor_params: FrozenVariableDict,
        critic_params: FrozenVariableDict,
    ):
        def env_step(collection_state: TrajectoryCollectionState, step_index: jax.Array):
            # Get keys for sampling action and stepping environment
            rng_steps = jax.random.fold_in(rng, 0)
            rng_action = jax.random.fold_in(rng, 1)

            rng_steps = jax.random.split(rng_steps, self.num_envs)

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
            next_obs, env_state, reward, done, _ = self.vmap_step(
                rng_steps, collection_state.env_state, action, self.env_params
            )

            if self.normalize_observations:
                rms_state, next_obs = self.update_and_normalize(
                    collection_state.rms_state, next_obs
                )
                collection_state = collection_state.replace(rms_state=rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                collection_state.last_obs, unclipped_action, log_prob, reward, value, done
            )
            collection_state = collection_state.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=collection_state.global_step + self.num_envs,
            )
            return collection_state, transition

        collection_state, trajectories = jax.lax.scan(
            env_step, collection_state, xs=jnp.arange(self.num_steps), length=self.num_steps
        )
        return collection_state, trajectories

    # NOTE: Changed the signature vs update_actor: not accepts/returns the actor TrainState.
    def update_actor_only(self, actor_ts: TrainState, batch: AdvantageMinibatch):
        def _actor_loss_fn(params: FrozenVariableDict):
            log_prob, entropy = self.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            assert isinstance(entropy, jax.Array)
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - self.ent_coef * entropy

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(
            actor_ts.params,
            actor=self.actor,
            batch=batch,
            clip_eps=self.clip_eps,
            ent_coef=self.ent_coef,
        )
        assert isinstance(actor_loss, jax.Array)

        # TODO: to log the loss here?
        return actor_ts.apply_gradients(grads=grads), actor_loss

    def update_critic_only(self, critic_ts: TrainState, batch: AdvantageMinibatch):
        def critic_loss_fn(params: FrozenVariableDict):
            value = self.critic.apply(params, batch.trajectories.obs)
            assert isinstance(value, jax.Array)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            assert isinstance(value_pred_clipped, jax.Array)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        critic_loss, grads = jax.value_and_grad(critic_loss_fn)(critic_ts.params)
        assert isinstance(critic_loss, jax.Array)
        return critic_ts.apply_gradients(grads=grads), critic_loss

    def update(self, ts: PPOTrainState, batch: AdvantageMinibatch):
        actor_ts, actor_loss = self.update_actor_only(ts.actor_ts, batch)
        critic_ts, critic_loss = self.update_critic_only(ts.critic_ts, batch)
        # ts, actor_loss = self.update_actor(ts, batch)
        # ts, critic_loss = self.update_critic(ts, batch)
        # return ts, (actor_loss, critic_loss)
        return ts.replace(actor_ts=actor_ts, critic_ts=critic_ts), (actor_loss, critic_loss)


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
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        super().on_fit_start(trainer, pl_module)
        self.total_transitions = 0
        self.total_episodes = 0
        self._start = time.perf_counter()

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: EpisodeData,
        batch_index: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_index)
        episodes = batch

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
        print(f"Steps per second: {steps_per_second}")
        logger.info(f"Steps per second: {steps_per_second}")
        logger.info(f"Episodes per second: {episodes_per_second}")
        logger.info(f"Updates per second: {updates_per_second}")

    @override
    def get_num_samples(self, batch: EpisodeData) -> int:
        if isinstance(batch, int):  # fixme
            return 1
        return int(np.prod(batch.obs.shape[:2]).item())

    @override
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_end(trainer, pl_module)


def main():
    env_id = "Pendulum-v1"

    from brax.envs import _envs as brax_envs
    from rejax.compat.brax2gymnax import create_brax

    if env_id in brax_envs:
        env, env_params = create_brax(
            env_id,
            episode_length=1000,
            action_repeat=1,
            auto_reset=True,
            batch_size=None,
        )
    else:
        env, env_params = gymnax.make(env_id=env_id)

    algo = JaxRlExample(env=env, env_params=env_params)
    from lightning.pytorch.loggers.csv_logs import CSVLogger

    # todo: number of epochs:
    num_evals = np.ceil(algo.learner.total_timesteps / algo.learner.eval_freq).astype(int)
    trainer = lightning.Trainer(
        max_epochs=num_evals, logger=CSVLogger(save_dir="logs/jax_rl_debug")
    )
    trainer.fit(algo)
    metrics = trainer.validate(algo)
    print(metrics)
    return


if __name__ == "__main__":
    main()
    exit()

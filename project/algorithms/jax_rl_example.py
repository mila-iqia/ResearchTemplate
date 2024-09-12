"""Idea: Use a custom 'trainer' or 'task function' to not use pytorch-lightning at all and do end-to-end Jax training."""

import functools
import logging
from datetime import datetime

import brax.envs
import jax
import matplotlib.pyplot as plt

# BUG! We get cuda_dnn.cc:535] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR if we don't import pytorch before jax!
import torch  # noqa
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

logger = logging.getLogger(__name__)


class JaxTrainer:
    def __init__(self, *args, **kwargs):
        logger.debug(f"Ignoring {args} and {kwargs} for now.")

    def fit(self):
        logger.debug("Fitting the model.")


def brax_ppo_train():
    env_name = "halfcheetah"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = "positional"  # @param ['generalized', 'positional', 'spring']
    train_fn = {
        "inverted_pendulum": functools.partial(
            ppo.train,
            num_timesteps=2_000_000,
            num_evals=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "inverted_double_pendulum": functools.partial(
            ppo.train,
            num_timesteps=20_000_000,
            num_evals=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "ant": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=10,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=4096,
            batch_size=2048,
            seed=1,
        ),
        "humanoid": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=10,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "reacher": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=4,
            unroll_length=50,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=256,
            max_devices_per_host=8,
            seed=1,
        ),
        "humanoidstandup": functools.partial(
            ppo.train,
            num_timesteps=100_000_000,
            num_evals=20,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=15,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=6e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "hopper": functools.partial(
            sac.train,
            num_timesteps=6_553_600,
            num_evals=20,
            reward_scaling=30,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            discounting=0.997,
            learning_rate=6e-4,
            num_envs=128,
            batch_size=512,
            grad_updates_per_step=64,
            max_devices_per_host=1,
            max_replay_size=1048576,
            min_replay_size=8192,
            seed=1,
        ),
        "walker2d": functools.partial(
            sac.train,
            num_timesteps=7_864_320,
            num_evals=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            discounting=0.997,
            learning_rate=6e-4,
            num_envs=128,
            batch_size=128,
            grad_updates_per_step=32,
            max_devices_per_host=1,
            max_replay_size=1048576,
            min_replay_size=8192,
            seed=1,
        ),
        "halfcheetah": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=20,
            reward_scaling=1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=512,
            seed=3,
        ),
        "pusher": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=30,
            num_minibatches=16,
            num_updates_per_batch=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=512,
            seed=3,
        ),
    }[env_name]

    max_y = {
        "ant": 8000,
        "halfcheetah": 8000,
        "hopper": 2500,
        "humanoid": 13000,
        "humanoidstandup": 75_000,
        "reacher": 5,
        "walker2d": 5000,
        "pusher": 0,
    }[env_name]
    min_y = {"reacher": -100, "pusher": -150}.get(env_name, 0)

    env = brax.envs.get_environment(env_name=env_name, backend=backend)

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        # clear_output(wait=True)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    plt.xlim([0, train_fn.keywords["num_timesteps"]])
    plt.ylim([min_y, max_y])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.plot(xdata, ydata)
    plt.show()

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


def rejax_ppo_train():
    from rejax import PPO

    # Get train function and initialize config for training
    algo = PPO.create(env="CartPole-v1", learning_rate=0.001)

    # Jit the training function
    train_fn = jax.jit(algo.train)

    # Vmap training function over 300 initial seeds
    vmapped_train_fn = jax.vmap(train_fn)

    # Train 300 agents!
    keys = jax.random.split(jax.random.PRNGKey(0), 300)
    train_state, evaluation = vmapped_train_fn(keys)

    # class JaxRlExample(PPO):
    #     def update(self, ts, batch):
    #         return super().update(ts, batch)


if __name__ == "__main__":
    brax_ppo_train()
    rejax_ppo_train()
    # JaxRlExample().update(1, 2)
    # JaxTrainer().fit()

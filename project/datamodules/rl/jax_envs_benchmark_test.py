import time
from collections.abc import Sequence

import gymnax
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnax.wrappers.gym import GymnaxToVectorGymWrapper
from pytest_benchmark.fixture import BenchmarkFixture
from torch_jax_interop import jax_to_torch, torch_to_jax


def bench_jax_to_torch_test(device: torch.device, benchmark: BenchmarkFixture):
    n_back_and_forths = 100

    def _back_and_forth_loop(n_back_and_forths: int = 1, with_copy: bool = False):
        start = time.perf_counter()
        for n in range(n_back_and_forths):
            v = torch.rand(100, 100, 100, device=device)
            if with_copy:
                v = v.clone()
            v = torch_to_jax(v)
            v = jax_to_torch(v)
        return time.perf_counter() - start

    time_taken = benchmark(_back_and_forth_loop, n_back_and_forths)
    print(f"Time taken for {n_back_and_forths=} between jax and torch: {time_taken}")


def steps_per_second(env_id: str, num_envs: int, total_steps: int, with_to_torch: bool = False):
    # Instantiate the environment & its settings.
    gymnax_env, env_params = gymnax.make(env_id)
    num_steps = 0
    start_time = time.perf_counter()
    env = GymnaxToVectorGymWrapper(gymnax_env, num_envs=num_envs, params=env_params, seed=123)
    from project.datamodules.rl.envs.gymnax import GymnaxVectorEnvToTorchWrapper

    if with_to_torch:
        env = GymnaxVectorEnvToTorchWrapper(env)
    obs = env.reset()
    episodes = 0
    while num_steps < total_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        num_steps += obs.shape[0]
        episodes += terminated.sum()
    time_delta = time.perf_counter() - start_time
    sps = num_steps / time_delta
    print(
        f"Num_envs: {num_envs}, Steps per second: {sps:.2f}, Time: {time_delta:.2f}s, steps: {num_steps}"
    )
    return sps


def compare_steps_per_seconds_jax_vectorenv(
    env_id: str = "CartPole-v1",
    num_envs: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
    total_steps: Sequence[int] | int | None = None,
):
    fig, ax = plt.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)

    if total_steps is None:
        # Set the total number of of steps as a multiple of the number of environments, so that we
        # don't wait for a long time with a low number of envs, and so we can amortize any higher
        # setup cost of setting up a larger vectorized env.
        total_steps_per_env = 5_000 * np.array(num_envs)
    elif isinstance(total_steps, int):
        # Use the same number of steps in all cases.
        assert total_steps > 1
        total_steps_per_env = np.full(len(num_envs), total_steps)
    else:
        assert len(total_steps) == len(num_envs)
        total_steps_per_env = np.array(total_steps)

    for with_to_torch in [True, False]:
        spss = [
            steps_per_second(
                env_id=env_id,
                num_envs=num_envs_i,
                total_steps=total_steps_i,
                with_to_torch=with_to_torch,
            )
            for num_envs_i, total_steps_i in zip(num_envs, total_steps_per_env)
        ]
        ax.scatter(num_envs, spss, label="jax->torch" if with_to_torch else "jax")
    fig.legend()
    ax.set_xlabel("Number of vectorized environments")
    ax.set_ylabel("Environment steps per second")
    ax.set_title(
        "Environment throughput (random actor), jax environments (wrapped with Torch or not)."
    )
    plt.ticklabel_format(style="plain", axis="y")
    fig.savefig("sps_hist.png")
    fig.show()
    fig.waitforbuttonpress()


# plt.waitforbuttonpress()
# # Reset the environment.
# obs, state = env.reset(key_reset, env_params)

# # Sample a random action.
# action = env.action_space(env_params).sample(key_act)

# # Perform the step transition.
# n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
# print(n_obs, n_state, reward, done)
# #
if __name__ == "__main__":
    compare_steps_per_seconds_jax_vectorenv()

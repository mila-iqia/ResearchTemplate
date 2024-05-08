import pytest

from project.datamodules.rl.envs.env_tests import EnvTests

# TODO: Run test with all gymnax envs:
# from gymnax.registration import registered_envs

# envs = registered_envs
envs = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
]


@pytest.mark.parametrize("env_id", envs, indirect=True)
class TestGymnaxEnv(EnvTests):
    """Tests for the wrapped Gymnax environments."""

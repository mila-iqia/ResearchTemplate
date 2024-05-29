import pytest

from project.datamodules.rl.envs.env_tests import EnvTests
from project.datamodules.rl.envs.gymnax import GymnaxVectorEnvToTorchWrapper

# TODO: Run test with all gymnax envs:
# from gymnax.registration import registered_envs

# envs = registered_envs
envs = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("env_id", envs, indirect=True)
class TestGymnaxEnv(EnvTests):
    """Tests for the wrapped Gymnax environments."""

    @pytest.mark.xfail(
        reason="TODO: Try to get the final observation and info from the gymnas envs somehow."
    )
    def test_vectorenv_info_on_episode_end(
        self, vectorenv: GymnaxVectorEnvToTorchWrapper, seed: int
    ):
        super().test_vectorenv_info_on_episode_end(vectorenv=vectorenv, seed=seed)

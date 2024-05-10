import pytest

from project.datamodules.rl.envs.env_tests import EnvTests

from .brax import BraxToTorchVectorEnv

# TODO: Run test with all brax envs:
# from project.datamodules.rl.envs.env_tests import EnvTests
# envs = list(brax.envs._envs.keys())
envs = [
    "halfcheetah",
]


@pytest.mark.parametrize("env_id", envs, indirect=True)
class TestBraxEnv(EnvTests):
    """Tests for the wrapped Brax environments."""

    @pytest.mark.xfail(
        reason="It seems almost impossible to get the final observation and info from the brax envs."
    )
    def test_vectorenv_info_on_episode_end(self, vectorenv: BraxToTorchVectorEnv, seed: int):
        super().test_vectorenv_info_on_episode_end(vectorenv=vectorenv)

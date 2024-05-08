import pytest

from project.datamodules.rl.envs.env_tests import EnvTests

# TODO: Run test with all brax envs:
# from project.datamodules.rl.envs.env_tests import EnvTests
# envs = list(brax.envs._envs.keys())
envs = [
    "halfcheetah",
]


@pytest.mark.parametrize("env_id", envs, indirect=True)
class TestBraxEnv(EnvTests):
    """Tests for the wrapped Brax environments."""

from typing import ClassVar

from project.algorithms.ppo.ppo import PPO
from project.algorithms.rl_example.reinforce_test import TestReinforce as ReinforceTests
from project.utils.testutils import (
    get_all_datamodule_names,
    get_all_network_names,
)


class TestPpo(ReinforceTests):
    algorithm_type: type[PPO] = PPO
    algorithm_name: ClassVar[str] = "ppo"

    unsupported_datamodule_names: ClassVar[list[str]] = list(
        set(get_all_datamodule_names()) - {"pendulum"}
    )
    unsupported_network_names: ClassVar[list[str]] = list(set(get_all_network_names()) - {"fcnet"})

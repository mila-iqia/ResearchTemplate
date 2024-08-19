from typing import ClassVar

from project.algorithms.ppo.ppo import PPO
from project.algorithms.rl_example.reinforce_test import TestReinforce as ReinforceTests
from project.datamodules.rl.datamodule import RlDataModule
from project.networks.fcnet import FcNet
from project.utils.testutils import run_for_all_configs_of_type


@run_for_all_configs_of_type("datamodule", RlDataModule)
@run_for_all_configs_of_type("network", FcNet)
class TestPpo(ReinforceTests):
    algorithm_type: type[PPO] = PPO
    algorithm_name: ClassVar[str] = "ppo"

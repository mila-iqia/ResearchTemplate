
import torch

from project.algorithms.ppo.ppo import PPO
from project.algorithms.testsuites.algorithm_tests import LearningAlgorithmTests
from project.datamodules.rl.datamodule import RlDataModule
from project.utils.testutils import run_for_all_configs_of_type


@run_for_all_configs_of_type("datamodule", RlDataModule)
@run_for_all_configs_of_type("network", torch.nn.Module)
class TestPpo(LearningAlgorithmTests[PPO]):
    ...

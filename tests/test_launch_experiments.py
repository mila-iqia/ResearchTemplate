import sys
import uuid

import pytest
from _pytest.mark.structures import ParameterSet
from omegaconf import DictConfig

import project.main
from project.conftest import command_line_overrides
from project.main import main
from project.main_test import experiment_commands_to_test, experiment_configs
from project.utils.testutils import IN_GITHUB_CI


@pytest.mark.slow
@pytest.mark.parametrize(
    command_line_overrides.__name__,
    experiment_commands_to_test,
    indirect=True,
)
def test_can_run_experiment(
    command_line_overrides: tuple[str, ...],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    # Mock out some part of the `main` function to not actually run anything.
    # Get a unique hash id:
    # todo: Set a unique name to avoid collisions between tests and reusing previous results.
    name = f"{request.function.__name__}_{uuid.uuid4().hex}"
    command_line_args = ["project/main.py"] + list(command_line_overrides) + [f"name={name}"]
    print(command_line_args)
    monkeypatch.setattr(sys, "argv", command_line_args)
    project.main.main()


@pytest.mark.parametrize("experiment_config", experiment_configs)
def test_experiment_config_is_tested(experiment_config: str):
    select_experiment_command = f"experiment={experiment_config}"

    for test_command in experiment_commands_to_test:
        if isinstance(test_command, ParameterSet):
            assert len(test_command.values) == 1
            assert isinstance(test_command.values[0], str), test_command.values
            test_command = test_command.values[0]

        if select_experiment_command in test_command:
            return  # success.

    pytest.fail(
        f"Experiment config {experiment_config!r} is not covered by any of the tests!\n"
        f"Consider adding an example of an experiment command that uses this config to the "
        # This is a 'nameof' hack to get the name of the variable so we don't hard-code it.
        + ("`" + f"{experiment_commands_to_test=}".partition("=")[0] + "` list")
        + " list.\n"
        f"For example: 'experiment={experiment_config} trainer.max_epochs=1'."
    )


# todo: kind-of redundant, given the other test commands.
@pytest.mark.skipif(
    IN_GITHUB_CI and sys.platform == "darwin",
    reason="TODO: Getting a 'MPS backend out of memory' error on the Github CI. ",
)
@pytest.mark.parametrize(
    command_line_overrides.__name__,
    [
        "algorithm=image_classifier datamodule=cifar10 seed=1 trainer/callbacks=none trainer.fast_dev_run=True"
    ],
    indirect=True,
)
def test_fast_dev_run(experiment_dictconfig: DictConfig):  # noqa
    result = main(experiment_dictconfig)
    assert isinstance(result, dict)
    assert result["type"] == "objective"
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], float)


# TODO: Add some more integration tests:
# - running sweeps from Hydra!
# - using the slurm launcher!
# - Test offline mode for narval and such.

from hydra_auto_schema.__main__ import main

from project.configs.config_test import CONFIG_DIR


def test_run_via_cli_without_errors():
    """Checks that the command completes without errors."""
    # Run programmatically instead of with a subprocess so we can get nice coverage stats.
    # assuming we're at the project root directory.
    main([str(CONFIG_DIR), "--stop-on-error"])

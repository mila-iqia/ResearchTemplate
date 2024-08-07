import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--shorter-than",
        action="store",
        type=float,
        default=None,
        help="Skip tests that take longer than this.",
    )


def pytest_ignore_collect(path: str):
    p = Path(path)
    # fixme: Trying to fix doctest issues for project/configs/algorithm/lr_scheduler/__init__.py::project.configs.algorithm.lr_scheduler.StepLRConfig
    if p.name in ["lr_scheduler", "optimizer"] and "configs" in p.parts:
        return True
    return False


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fast: mark test as fast to run (after fixtures are setup)")
    config.addinivalue_line(
        "markers", "very_fast: mark test as very fast to run (including test setup)."
    )

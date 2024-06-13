import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--shorter-than",
        action="store",
        type=float,
        default=None,
        help="Skip tests that take longer than this.",
    )


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fast: mark test as fast to run (after fixtures are setup)")
    config.addinivalue_line(
        "markers", "very_fast: mark test as very fast to run (including test setup)."
    )

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark-marked tests.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-benchmarks"):
        return

    skip_benchmark = pytest.mark.skip(
        reason="benchmark tests are opt-in (pass --run-benchmarks to run them)",
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)

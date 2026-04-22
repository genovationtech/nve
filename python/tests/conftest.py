"""pytest configuration for NVE Python SDK tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (requires pytest-asyncio)"
    )

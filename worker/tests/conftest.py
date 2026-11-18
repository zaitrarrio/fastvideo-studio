import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1]
REPO = WORKER.parent
sys.path.insert(0, str(WORKER))

from fvstudio.catalog import Catalog  # noqa: E402


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog.load(REPO / "catalog" / "catalog.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: builds the engines and runs the synthetic serves (STUDIO_E2E=1)")

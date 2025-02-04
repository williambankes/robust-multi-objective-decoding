import pytest
import pathlib


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent

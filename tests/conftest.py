import socket
from pathlib import Path

import pytest

from setsmith.io.rekordbox_xml import load_collection
from setsmith.model.collection import Collection

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own Setsmith database, never the user's real one."""
    db = tmp_path / "setsmith-test.db"
    monkeypatch.setenv("SETSMITH_DB", str(db))
    return db


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never reach real services: any outgoing connection fails the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"a test tried to open a network connection: {args[:1]}")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def fixture_5() -> Path:
    return FIXTURES / "collection_5.xml"


@pytest.fixture
def fixture_50() -> Path:
    return FIXTURES / "collection_50.xml"


@pytest.fixture
def collection_5(fixture_5: Path) -> Collection:
    return load_collection(fixture_5)


@pytest.fixture
def collection_50(fixture_50: Path) -> Collection:
    return load_collection(fixture_50)


@pytest.fixture(scope="session")
def fixture_pool(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """300 synthetic house tracks at 118-128 BPM, energies 1-10, some missing data."""
    from synth import random_tracks, write_collection

    path = tmp_path_factory.mktemp("pool") / "pool_300.xml"
    write_collection(path, random_tracks(300, seed=3, bpm_ranges=((118.0, 128.0),)))
    return path


@pytest.fixture(scope="session")
def collection_pool(fixture_pool: Path) -> Collection:
    return load_collection(fixture_pool)

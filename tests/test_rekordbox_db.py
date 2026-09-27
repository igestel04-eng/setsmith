"""Optional master.db import (My Tags, history) against a fake, unencrypted database."""

import hashlib
import json
import unicodedata
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("pyrekordbox")

from fake_rekordbox import make_master_db
from synth import SynthTrack, write_collection
from typer.testing import CliRunner

import setsmith.cli as cli
from setsmith.cli import app
from setsmith.io.enrich import apply_my_tags, tag_names
from setsmith.io.rekordbox_db import (
    RekordboxDbData,
    RekordboxDbError,
    copy_master_db,
    normalize_path,
    read_master_db,
)
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.store import Store

runner = CliRunner()
FAKE_KEY = "402fd" + "0" * 59  # shape of a real key; the fakes are unencrypted anyway


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def library(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """A fake Rekordbox folder (master.db + -wal) and an XML export pointing at the same files."""
    music = tmp_path / "Music"
    music.mkdir()
    files = {
        "1": music / "Café Vocal.mp3",
        "2": music / "Dub Tool.mp3",
        "3": music / "Untagged.mp3",
    }
    for f in files.values():
        f.write_bytes(b"x")
    rb = tmp_path / "Pioneer" / "rekordbox"
    rb.mkdir(parents=True)
    master = make_master_db(
        rb / "master.db",
        # master.db stores decomposed accents, as macOS paths can be
        {cid: unicodedata.normalize("NFD", str(f)) for cid, f in files.items()},
        {
            "1": [("Components", "Vocal"), ("Situation", "Opener")],
            "2": [("Components", "Instrumental"), ("Situation", "Peak")],
        },
        [("Friday", "2026-09-20", ["2", "1", "3"]), ("Saturday", "2026-09-21", ["1", "2"])],
    )
    (rb / "master.db-wal").write_bytes(b"")
    xml = tmp_path / "rekordbox.xml"
    write_collection(
        xml,
        [
            SynthTrack(cid, "Artist " + cid, f.stem, 124.0, "Am", "House", 5, 3,
                       location="file://localhost" + quote(str(f)))
            for cid, f in files.items()
        ],
    )  # fmt: skip
    return master, xml, files


def fake_reader(path: Path, key: str | None) -> RekordboxDbData:
    assert key == FAKE_KEY
    return read_master_db(path, key, unlock=False)


def test_read_tags_and_history(library: tuple[Path, Path, dict[str, Path]]) -> None:
    master, _, files = library
    data = read_master_db(master, None, unlock=False)
    assert data.content_count == 3
    vocal_path = normalize_path(files["1"])
    assert data.my_tags[vocal_path] == ["Components: Vocal", "Situation: Opener"]
    assert [s.name for s in data.sessions] == ["Friday", "Saturday"]
    assert data.sessions[0].paths == [normalize_path(files[c]) for c in ("2", "1", "3")]


def test_key_required_and_wrong_key(library: tuple[Path, Path, dict[str, Path]]) -> None:
    master, _, _ = library
    with pytest.raises(RekordboxDbError, match="key is required"):
        read_master_db(master, None)
    with pytest.raises(RekordboxDbError, match="could not"):
        read_master_db(master, FAKE_KEY)  # a plain file opened as encrypted


def test_copy_never_touches_source_and_prunes(
    library: tuple[Path, Path, dict[str, Path]], tmp_path: Path
) -> None:
    master, _, _ = library
    before = (sha(master), master.stat().st_mtime)
    copies = tmp_path / "copies"
    made = [copy_master_db(master, copies, keep=2) for _ in range(3)]
    assert (sha(master), master.stat().st_mtime) == before
    assert made[-1].with_name("master.db-wal").exists()  # sidecar copied too
    remaining = sorted(d for d in copies.iterdir())
    assert remaining == sorted(p.parent for p in made[1:])  # oldest pruned
    unrelated = copies / "not-ours"
    unrelated.mkdir()
    copy_master_db(master, copies, keep=1)
    assert unrelated.exists()  # only Setsmith's own copies are pruned
    with pytest.raises(RekordboxDbError, match="not found"):
        copy_master_db(tmp_path / "nope.db", copies, keep=1)


def test_store_round_trip(library: tuple[Path, Path, dict[str, Path]], tmp_path: Path) -> None:
    master, _, _ = library
    data = read_master_db(master, None, unlock=False)
    with Store(tmp_path / "s.db") as store:
        assert store.rekordbox_import_info() is None
        store.save_rekordbox_data(data, str(master))
        store.save_rekordbox_data(data, str(master))  # a re-import replaces, not duplicates
        assert store.load_my_tags() == data.my_tags
        history = store.load_history()
        assert [(s.name, s.paths) for s in history] == [(s.name, s.paths) for s in data.sessions]
        info = store.rekordbox_import_info()
        assert info is not None and info["tracks_tagged"] == 2 and info["sessions"] == 2


def test_apply_my_tags_sets_vocal() -> None:
    col = Collection(
        tracks={
            "1": Track(id="1", title="Song (Dub)", location="file://localhost/m/a.mp3"),
            "2": Track(id="2", location="file://localhost/m/b.mp3"),
        }
    )
    col.tracks["1"].vocal = False  # from the "(Dub)" title
    cov = apply_my_tags(col, {"/m/a.mp3": ["Components: Vocal"], "/m/b.mp3": ["Vibe: Dark"]})
    assert col.tracks["1"].vocal is True  # an explicit My Tag wins over title words
    assert col.tracks["2"].vocal is None and col.tracks["2"].my_tags == ["Vibe: Dark"]
    assert (cov.tagged, cov.vocal_from_tags) == (2, 1)
    assert tag_names(col.tracks["2"]) == {"vibe: dark", "dark"}


def test_cli_import_flow(
    library: tuple[Path, Path, dict[str, Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    master, xml, _ = library
    monkeypatch.setattr(cli, "read_master_db", fake_reader)
    monkeypatch.setattr(cli, "rekordbox_running", lambda: False)
    base = ["rekordbox", "import", "--master-db", str(master)]

    no_backup = runner.invoke(app, [*base, "--key", FAKE_KEY])
    assert no_backup.exit_code == 2 and "Backup Library" in no_backup.output
    no_key = runner.invoke(app, [*base, "--backed-up"], env={"SETSMITH_REKORDBOX_KEY": ""})
    assert no_key.exit_code == 2

    before = sha(master)
    result = runner.invoke(
        app, [*base, "--backed-up", "--json"], env={"SETSMITH_REKORDBOX_KEY": FAKE_KEY}
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["tracks_tagged"] == 2 and data["sessions"] == 2
    assert data["my_tags"]["Situation: Opener"] == 1
    assert sha(master) == before
    assert Path(data["copy"]).parent.parent.name == "rekordbox-copies"

    status = json.loads(runner.invoke(app, ["rekordbox", "status", "--json"]).output)
    assert status["tracks_tagged"] == 2

    info = json.loads(runner.invoke(app, ["info", str(xml), "--json"]).output)
    assert info["my_tags"] == {"tagged": 2, "vocal_from_tags": 2}

    built = runner.invoke(
        app, ["build", str(xml), "--tracks", "2", "--tag", "opener", "--tag", "Peak", "--json"]
    )
    assert built.exit_code == 0, built.output
    ids = {p["track"]["id"] for p in json.loads(built.output)["positions"]}
    assert ids == {"1", "2"}  # the untagged track is filtered out
    tags = {p["track"]["id"]: p["track"]["my_tags"] for p in json.loads(built.output)["positions"]}
    assert tags["1"] == ["Components: Vocal", "Situation: Opener"]


def test_cli_status_before_import() -> None:
    result = runner.invoke(app, ["rekordbox", "status"])
    assert result.exit_code == 0 and "Nothing imported yet" in result.output

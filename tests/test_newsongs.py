"""Building sets with songs outside the library (fake services, no network)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import setsmith.cli as cli
import setsmith.discovery.newsongs as newsongs
from setsmith.discovery.discover import DiscoveryUnavailable
from setsmith.discovery.newsongs import NEW_PREFIX, NewSongs, gather_new_songs, with_tracks
from setsmith.discovery.song import EXTERNAL_ID, Song
from setsmith.io.rekordbox_xml import load_collection
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import SetRequest, generate_set
from setsmith.sets.report import render_markdown, set_to_dict

runner = CliRunner()
LINKS = {
    "soundcloud": "https://soundcloud.com/search/sounds?q=x",
    "lastfm": "https://www.last.fm/a_(b)",
}


def found(title: str, artist: str, bpm: float | None = 121.0, off_scene: bool = False) -> Any:
    track = Track("discovered:1", title, artist, genre="house", bpm=bpm, camelot="8A",
                  key_confidence=0.6)  # fmt: skip
    return SimpleNamespace(track=track, off_scene=off_scene, links=LINKS,
                           bpm_key_source="estimated from the Deezer preview")  # fmt: skip


@pytest.fixture
def fake_discovery(monkeypatch: pytest.MonkeyPatch, collection_5: Collection) -> None:
    outside = Track(EXTERNAL_ID, "Outside", "Zed", genre="house", bpm=121.0, camelot="8A")
    songs = {
        "Kaya Sol - Sunwater": Song(collection_5.tracks["101"], in_library=True),
        "Zed - Outside": Song(
            outside, in_library=False, bpm_key_source="GetSongBPM", links=dict(LINKS)
        ),
    }
    results = {
        "101": [
            found("New Tide", "Moana Reyes"),  # the library has Moana Reyes as Afro House
            found("Velvet Drift", "Tomas Lind"),  # unknown artist: the picked song's genre
            found("Anthem", "Big Room Hero", off_scene=True),  # another scene: left out
            found("No Tempo", "Somebody", bpm=None),  # the builder needs a BPM
        ],
        EXTERNAL_ID: [found("Velvet Drift", "Tomas Lind")],  # already found: once only
    }
    monkeypatch.setattr(newsongs, "make_clients", lambda store, cfg: (object(), None, None))
    monkeypatch.setattr(newsongs, "resolve_song", lambda q, *a, **k: songs[q])
    monkeypatch.setattr(
        newsongs, "discover", lambda col, seed, *a, **k: SimpleNamespace(items=results[seed.id])
    )


@pytest.mark.usefixtures("fake_discovery")
def test_gather_new_songs(collection_5: Collection) -> None:
    new = gather_new_songs(["Kaya Sol - Sunwater", "Zed - Outside"], collection_5, None)
    got = {(t.artist, t.title): t for t in new.tracks}
    assert set(got) == {("Moana Reyes", "New Tide"), ("Tomas Lind", "Velvet Drift"),
                        ("Zed", "Outside")}  # fmt: skip
    assert all(t.id.startswith(NEW_PREFIX) for t in new.tracks)
    assert got["Moana Reyes", "New Tide"].genre == "Afro House"  # your genre for the artist
    assert got["Tomas Lind", "Velvet Drift"].genre == "Afro House"  # the picked song's
    assert got["Zed", "Outside"].genre == "house"  # nothing better known
    zed = new.outside[got["Zed", "Outside"].id]
    assert zed["found_via"] == "one of your picked songs" and zed["links"] == LINKS
    assert any("around Zed - Outside" in w for w in new.warnings)  # nothing new around it


@pytest.mark.usefixtures("fake_discovery")
def test_gather_limits(collection_5: Collection, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="at least one"):
        gather_new_songs([" "], collection_5, None)
    with pytest.raises(ValueError, match="at most 5"):
        gather_new_songs(["a - b"] * 6, collection_5, None)
    monkeypatch.setattr(newsongs, "make_clients", lambda store, cfg: (None, None, None))
    with pytest.raises(DiscoveryUnavailable):
        gather_new_songs(["Kaya Sol - Sunwater"], collection_5, None)


def pool(n_new: int) -> tuple[Collection, frozenset[str]]:
    """Owned and new tracks that all mix well (same tempo and key, distinct artists)."""

    def track(tid: str, title: str, artist: str, confidence: float = 1.0) -> Track:
        return Track(tid, title, artist, genre="Afro House", bpm=122.0, camelot="8A",
                     key_confidence=confidence)  # fmt: skip

    owned = [track(f"{i}", f"Owned {i}", f"Artist {i}") for i in range(1, 9)]
    new = [
        track(f"{NEW_PREFIX}{i}", f"New {i}", f"New Artist {i}", 0.6) for i in range(1, n_new + 1)
    ]
    col = with_tracks(Collection({t.id: t for t in owned}), new)
    return col, frozenset(t.id for t in new)


@pytest.mark.parametrize(("share", "expected"), [(0.0, 0), (0.25, 2), (0.5, 4)])
def test_build_reaches_the_new_song_share(share: float, expected: int) -> None:
    col, new_ids = pool(8)
    request = SetRequest(curve=parse_curve("journey"), track_count=8, new_ids=new_ids,
                         max_new_share=share)  # fmt: skip
    gen, _ = generate_set(col, request)
    picked = [p.track.id in new_ids for p in gen.positions]
    assert sum(picked) == expected  # the share is a target and a cap
    if expected:
        assert picked != [True] * expected + [False] * (8 - expected)  # spread, not bunched


def test_report_lists_songs_to_get() -> None:
    col, new_ids = pool(2)
    gen, _ = generate_set(
        col, SetRequest(curve=parse_curve("journey"), track_count=6, new_ids=new_ids)
    )
    gen.outside = {tid: {"found_via": "similar to X", "links": LINKS} for tid in new_ids}
    text = render_markdown(gen, "Mixed")
    assert "## Songs to get" in text and "(new)" in text
    assert "[Last.fm](https://www.last.fm/a_%28b%29)" in text  # parentheses can't break it
    assert set(set_to_dict(gen, "Mixed")["outside"]) <= new_ids


def fake_gather(queries: list[str], collection: Collection, *a: Any, **k: Any) -> NewSongs:
    result = NewSongs()
    for i, artist in enumerate(["Tomas Lind", "Ines Okafor"], 1):
        track = Track("x", f"Fresh {i}", artist, genre="Afro House", bpm=122.0, camelot="8A")
        result.add(track, dict(LINKS), "similar to Kaya Sol - Sunwater", "GetSongBPM")
    return result


def test_cli_build_around(fixture_5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "gather_new_songs", fake_gather)
    out = tmp_path / "set.xml"
    args = ["build", str(fixture_5), "--around", "Kaya Sol - Sunwater", "-n", "4",
            "--new-share", "0.5", "-o", str(out), "--json"]  # fmt: skip
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    ids = [p["track"]["id"] for p in data["positions"]]
    new = [i for i in ids if i.startswith(NEW_PREFIX)]
    assert len(new) == 2 and set(new) <= set(data["outside"])
    exported = load_collection(out)
    assert exported.playlists is not None
    playlist = next(n for _, n in exported.playlists.walk() if n.name == data["name"])
    assert list(playlist.track_keys) == [i for i in ids if not i.startswith(NEW_PREFIX)]
    text = runner.invoke(cli.app, args[:-3])
    assert "Songs to get" in text.output and "(new)" in text.output


def test_web_build_around(
    fixture_5: Path, isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import setsmith.web.app as web

    monkeypatch.setattr(web, "gather_new_songs", fake_gather)
    client = TestClient(web.create_app(fixture_5, isolated_db))
    body = {"tracks": 4, "around": ["Kaya Sol - Sunwater"], "new_share": 0.5}
    built = client.post("/api/build", json=body).json()
    ids = [p["track"]["id"] for p in built["set"]["positions"]]
    assert sum(i.startswith(NEW_PREFIX) for i in ids) == 2
    assert set(built["set"]["outside"]) >= {i for i in ids if i.startswith(NEW_PREFIX)}
    export = client.get(f"/api/sets/{built['id']}/export")
    assert NEW_PREFIX.encode() not in export.content
    too_many = client.post("/api/build", json={"tracks": 4, "around": ["a - b"] * 6})
    assert too_many.status_code == 422

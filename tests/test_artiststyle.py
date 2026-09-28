"""Style profiles for any artist, with fake services (no network in tests)."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import setsmith.discovery.artiststyle as artiststyle
from setsmith.cli import app
from setsmith.discovery.artiststyle import (
    ESTIMATE_SOURCE,
    estimate_artist_style,
    save_artist_style,
)
from setsmith.discovery.http import ServiceError
from setsmith.discovery.lastfm import LastFmTrack
from setsmith.discovery.tempokey import TempoKey
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.styles.profile import load_style

runner = CliRunner()
# Popular tracks: mostly 120-122 BPM in minor keys, plus one slow outlier.
MEASURED = [(120.0, "8A"), (121.0, "5A"), (122.0, "4A"), (120.5, "8B"), (98.0, "1A"), (121.5, None)]


class FakeLastFm:
    def __init__(self, known: bool = True) -> None:
        self.known = known

    def top_tracks(self, artist: str, limit: int, match: float) -> list[LastFmTrack]:
        if not self.known:
            raise ServiceError("Last.fm: The artist you supplied could not be found (code 6)")
        return [LastFmTrack(artist, f"Hit {i}", "", match, "") for i in range(len(MEASURED))]

    def similar_artists(self, artist: str, limit: int) -> list[tuple[str, float]]:
        return [("Rampa", 1.0), ("&ME", 0.8)][:limit]

    def artist_tags(self, artist: str) -> list[str]:
        return ["house", "afro house"]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artiststyle, "make_clients", lambda store, cfg: (FakeLastFm(), None, None))

    def lookup(songs: list[tuple[str, str]], *a: Any, **k: Any) -> tuple[list[Any], list[str]]:
        return [
            TempoKey(b, c, 0.6, "estimated from the Deezer preview", {}) for b, c in MEASURED
        ], []

    monkeypatch.setattr(artiststyle, "lookup_tempo_key", lookup)


@pytest.mark.usefixtures("fakes")
def test_estimate_artist_style() -> None:
    style = estimate_artist_style("Mystery DJ", Collection({}), None)
    p = style.profile
    assert (p.name, p.kind) == ("Inspired by Mystery DJ", "artist")
    lo, hi = p.bpm_band
    assert 117 <= lo <= 120 and 122 <= hi <= 124  # the 98 BPM outlier is left out
    assert p.bpm_preferred == 121.0  # the median, 120.75, to the nearest half BPM
    assert p.key_mode_preference.minor == 0.8  # 4 of 5 known keys
    assert style.genre_style == "afro_house"  # from Last.fm's tags
    assert p.energy_curve == load_style("afro_house").energy_curve  # habits of the genre
    assert p.reference_artists == ["Mystery DJ", "Rampa", "&ME"]
    assert (style.tempos, style.keys) == (6, 5)


@pytest.mark.usefixtures("fakes")
def test_your_genre_tags_win() -> None:
    owned = Track("1", "Old Hit", "Mystery DJ", genre="Tech House", bpm=126.0)
    style = estimate_artist_style("Mystery DJ", Collection({"1": owned}), None)
    assert style.genre_style == "tech_house"


@pytest.mark.usefixtures("fakes")
def test_estimate_needs_tempos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        artiststyle, "lookup_tempo_key", lambda songs, *a, **k: ([None] * len(songs), [])
    )
    with pytest.raises(ValueError, match="too few"):
        estimate_artist_style("Mystery DJ", Collection({}), None)
    monkeypatch.setattr(
        artiststyle, "make_clients", lambda s, c: (FakeLastFm(known=False), None, None)
    )
    with pytest.raises(ValueError, match="doesn't know"):
        estimate_artist_style("Nobody", Collection({}), None)


@pytest.mark.usefixtures("fakes")
def test_save_never_replaces_other_profiles(isolated_user_styles: Path) -> None:
    style = estimate_artist_style("Mystery DJ", Collection({}), None)
    key, path = save_artist_style(style)
    assert key == "mystery_dj" and path.parent == isolated_user_styles
    assert load_style("mystery_dj").sources == [ESTIMATE_SOURCE]
    assert save_artist_style(style)[0] == "mystery_dj"  # re-estimating replaces its own
    (isolated_user_styles / "mystery_dj.json").write_text(
        json.dumps({**json.loads(path.read_text()), "sources": ["my own notes"]})
    )
    assert save_artist_style(style)[0] == "mystery_dj_estimated"  # a hand-made one stays
    keinemusik = estimate_artist_style("Keinemusik", Collection({}), None)
    assert save_artist_style(keinemusik)[0] == "keinemusik_estimated"  # never a built-in
    assert load_style("keinemusik").sources != [ESTIMATE_SOURCE]


@pytest.mark.usefixtures("fakes")
def test_cli_styles_artist() -> None:
    result = runner.invoke(app, ["styles", "artist", "Mystery DJ", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["key"] == "mystery_dj" and data["genre_style"] == "afro_house"
    listing = json.loads(runner.invoke(app, ["styles", "list", "--json"]).output)
    assert {"key": "mystery_dj", "kind": "artist"}.items() <= next(
        e for e in listing if e["key"] == "mystery_dj"
    ).items()


@pytest.mark.usefixtures("fakes")
def test_web_artist_style(fixture_pool: Path, isolated_db: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import setsmith.web.app as web

    client = TestClient(web.create_app(fixture_pool, isolated_db))
    made = client.post("/api/styles/artist", json={"artist": "Mystery DJ"}).json()
    assert made["key"] == "mystery_dj" and made["kind"] == "artist"
    info = client.get("/api/info").json()
    assert "mystery_dj" in {s["key"] for s in info["styles"]}
    built = client.post("/api/build", json={"tracks": 3, "style": "mystery_dj"})
    assert built.status_code == 200, built.text
    assert client.post("/api/styles/artist", json={"artist": ""}).status_code == 422

"""Deezer preview estimates and "any song" seeds, with fake services (no network in tests)."""

import io
import json
import math
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import setsmith.cli as cli
import setsmith.discovery.song as song_module
from setsmith.discovery.getsongbpm import GetSongBpm
from setsmith.discovery.http import ServiceError
from setsmith.discovery.lastfm import LastFm
from setsmith.discovery.preview import DeezerPreviews, fold_bpm, strip_id3
from setsmith.discovery.song import (
    EXTERNAL_ID,
    library_ids,
    library_match,
    parse_song,
    resolve_song,
    with_track,
)
from setsmith.discovery.tempokey import DEEZER_PREVIEW, GETSONGBPM, lookup_tempo_key
from setsmith.io.rekordbox_xml import load_collection
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG

runner = CliRunner()
CDN = "https://cdnt-preview.dzcdn.net/api/1/1/a/b/c/0/abc.mp3"


class FakeClient:
    """Answers JsonClient.get from canned responses keyed by path (and query)."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, path: str, params: dict[str, str]) -> Any:
        self.calls.append((path, params))
        key = params.get("method") or path
        if key == "search":
            key = f"search:{params['q']}"
        if key == "track.search":
            key = f"track.search:{params['track']}"
        missing: dict[str, Any] = {"data": []} if path == "search" else {"error": 6}
        return self.responses.get(key, missing)


class MemoryCache:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def cache_get(self, key: str, max_age_s: float) -> str | None:
        return self.data.get(key)

    def cache_put(self, key: str, body: str) -> None:
        self.data[key] = body


def hit(i: int, artist: str, title: str, **extra: Any) -> dict[str, Any]:
    return {"id": i, "title": title, "artist": {"name": artist}, "preview": CDN,
            "link": f"https://www.deezer.com/track/{i}", **extra}  # fmt: skip


DEEZER = {
    "search:Rampa Crazy For It": {"data": [hit(1, "Rampa", "Crazy For It (Extended Mix)")]},
    # A different song with a similar title must not be taken.
    "search:&ME The Rapture Pt.II": {"data": [hit(2, "&ME", "The Rapture Pt.III")]},
    # Search results name only the main artist; the co-credit is on the track.
    "search:Black Coffee Drive": {
        "data": [hit(3, "David Guetta", "Drive (feat. Delilah Montagu)")]
    },
    "track/3": {"id": 3, "contributors": [{"name": "David Guetta"}, {"name": "Black Coffee"}]},
    "search:Tomas Lind Velvet Drift": {"data": [hit(4, "Tomas Lind", "Velvet Drift")]},
    "search:Kaya Sol Tidewater": {"data": [hit(5, "Kaya Sol", "Tidewater")]},
}  # fmt: skip


def fake_analyze(data: bytes, cfg: Any) -> tuple[float | None, str | None, float, str]:
    return 121.5, "8A", 0.9, "fake"


def previews(
    responses: dict[str, Any] | None = None,
    cache: MemoryCache | None = None,
    fetch: Any = None,
) -> tuple[DeezerPreviews, FakeClient, list[str]]:
    fetched: list[str] = []

    def fake_fetch(url: str, cfg: Any) -> bytes:
        fetched.append(url)
        return b"mp3"

    client = FakeClient(DEEZER if responses is None else responses)
    dp = DeezerPreviews(client, cache, analyze=fake_analyze, fetch=fetch or fake_fetch)  # type: ignore[arg-type]
    return dp, client, fetched


# ---------------------------------------------------------------- preview helpers


def test_strip_id3() -> None:
    tag = b"ID3\x04\x00\x00" + bytes([0, 0, 1, 2]) + b"x" * 130  # size 130 (syncsafe 1,2)
    assert strip_id3(tag + b"MP3DATA") == b"MP3DATA"
    footer = b"ID3\x04\x00\x10" + bytes([0, 0, 0, 5]) + b"x" * 15  # 5 bytes + 10-byte footer
    assert strip_id3(footer + b"MP3") == b"MP3"
    assert strip_id3(b"\xff\xfbMP3") == b"\xff\xfbMP3"


def test_vote_bpm() -> None:
    from setsmith.discovery.preview import vote_bpm

    assert vote_bpm([160.1, 119.9, 120.2], 2.0) == pytest.approx(120.05)  # two of three
    assert vote_bpm([121.0, 121.5, 120.8], 2.0) == pytest.approx(121.1)
    assert vote_bpm([93.9, 120.0, 161.5], 2.0) is None  # no two agree
    assert vote_bpm([120.0], 2.0) is None


@pytest.mark.parametrize(("bpm", "folded"), [(61.0, 122.0), (244.0, 122.0), (122.0, 122.0)])
def test_fold_bpm(bpm: float, folded: float) -> None:
    assert fold_bpm(bpm, 88.0) == folded


def test_deezer_find() -> None:
    dp, client, _ = previews()
    found = dp.find("Rampa", "Crazy For It")
    assert found is not None and found.deezer_id == 1  # "(Extended Mix)" is ignored
    assert dp.find("&ME", "The Rapture Pt.II") is None  # Pt.III is another song
    drive = dp.find("Black Coffee", "Drive - Edit")  # searched as "Drive"
    assert drive is not None and drive.deezer_id == 3
    assert ("search", {"q": "Black Coffee Drive", "limit": "5"}) in client.calls
    assert dp.find("Nobody", "Nothing") is None


def test_deezer_error() -> None:
    dp, _, _ = previews({"search:A B": {"error": {"message": "Quota limit exceeded", "code": 4}}})
    with pytest.raises(ServiceError, match="Quota"):
        dp.find("A", "B")


def test_lookup_many_caches_and_reports() -> None:
    cache = MemoryCache()
    dp, _, fetched = previews(cache=cache)
    songs = [
        ("Rampa", "Crazy For It"),
        ("&ME", "The Rapture Pt.II"),
        ("Tomas Lind", "Velvet Drift"),
    ]
    results, notes = dp.lookup_many(songs)
    assert [r.bpm if r else None for r in results] == [121.5, None, 121.5]
    assert len(fetched) == 2 and not notes
    repeat, repeat_client, repeat_fetched = previews(cache=cache)
    again, _ = repeat.lookup_many(songs)
    assert [r.camelot if r else None for r in again] == ["8A", None, "8A"]  # from the cache
    assert again[0] and again[0].hit.link == "https://www.deezer.com/track/1"
    # Cached songs need no Deezer request (search results hold expiring preview links).
    assert [c[1]["q"] for c in repeat_client.calls] == ["&ME The Rapture Pt.II"]
    assert not repeat_fetched

    def broken(url: str, cfg: Any) -> bytes:
        raise ServiceError("could not fetch a Deezer preview")

    failing = previews(fetch=broken)[0]
    results, notes = failing.lookup_many(songs[:1])
    assert results == [None] and "1 Deezer preview(s)" in notes[0]


def test_fetch_only_from_deezer_cdn() -> None:
    from setsmith.discovery.preview import _fetch_preview

    with pytest.raises(ServiceError, match="CDN"):
        _fetch_preview("https://evil.example/a.mp3", DEFAULT_CONFIG)
    with pytest.raises(ServiceError, match="CDN"):
        _fetch_preview("http://cdnt-preview.dzcdn.net/a.mp3", DEFAULT_CONFIG)


def test_analyze_audio_click_track(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("librosa")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    import setsmith.discovery.preview as preview

    sr, seconds, bpm = 44100, 20.0, 124.0
    t = np.arange(int(sr * seconds)) / sr
    chord = sum(np.sin(2 * math.pi * f * t) for f in (220.0, 261.63, 329.63)) * 0.1  # A minor
    clicks = np.zeros_like(t)
    for beat in np.arange(0, seconds, 60 / bpm):
        start = int(beat * sr)
        clicks[start : start + 400] = np.hanning(800)[400:] * 0.8
    buf = io.BytesIO()
    sf.write(buf, (chord + clicks).astype("float32"), sr, format="WAV")
    monkeypatch.setattr(preview, "has_module", lambda name: name != "essentia")  # librosa path
    est_bpm, key, _, method = preview.analyze_audio(buf.getvalue())
    assert method == "librosa"
    assert est_bpm is not None and abs(est_bpm - bpm) < 1.5
    assert key in {"8A", "5B"}  # A minor, or its relative C major


# ---------------------------------------------------------------- GetSongBPM then Deezer


class FakeBpm:
    def __init__(self, answers: dict[str, Any], fail: bool = False) -> None:
        self.answers, self.fail = answers, fail

    def lookup(self, artist: str, title: str) -> Any:
        if self.fail:
            raise ServiceError("GetSongBPM answered HTTP 401")
        return self.answers.get(title)


def test_lookup_tempo_key_chain() -> None:
    from setsmith.discovery.getsongbpm import SongInfo

    bpm = FakeBpm({
        "Crazy For It": SongInfo(120.0, "8A", "https://getsongbpm.com/song/x"),
        "Velvet Drift": SongInfo(121.0, None, ""),  # tempo only: Deezer fills in the key
    })  # fmt: skip
    dp, _, _ = previews()
    songs = [("Rampa", "Crazy For It"), ("Tomas Lind", "Velvet Drift"), ("Kaya Sol", "Tidewater")]
    results, _ = lookup_tempo_key(songs, bpm, dp)  # type: ignore[arg-type]
    first, second, third = results
    assert first and (first.bpm, first.source, first.key_confidence) == (120.0, GETSONGBPM, 1.0)
    assert second and (second.bpm, second.camelot, second.source) == (121.0, "8A", DEEZER_PREVIEW)
    assert second.key_confidence == DEFAULT_CONFIG.discovery.preview_key_confidence
    assert third and third.source == DEEZER_PREVIEW and "deezer" in third.links
    with pytest.raises(ServiceError):  # a bad key shows on the first request
        lookup_tempo_key(songs, FakeBpm({}, fail=True), dp)  # type: ignore[arg-type]


# ---------------------------------------------------------------- any song


def small_collection() -> Collection:
    tracks = [
        Track("1", "Adam Port, Stryv, Malachiii - Move (Extended)", "Adam Port", bpm=120.0,
              camelot="1A"),
        Track("2", "Adam Port, Stryv, Malachiii - Move (Extended)", "Adam Port\uff0c Stryv",
              bpm=120.0, camelot="1A"),  # a second upload of the same song
        Track("3", "Move Your Body (Future House)", "Tchami", bpm=124.0, camelot="9A"),
    ]  # fmt: skip
    return Collection({t.id: t for t in tracks})


def test_library_match_is_exact() -> None:
    col = small_collection()
    found = library_match(col, "Adam Port", "Move")
    assert found is not None and found.id in {"1", "2"}  # duplicates are one song
    assert library_match(col, "Tchami", "Move") is None  # not "Move Your Body"
    assert library_match(col, "Stryv", "Move") is not None  # a co-credited artist


def fake_lastfm() -> LastFm:
    client = FakeClient({
        "track.search:mwaki zerb": {"results": {"trackmatches": {"track": [
            {"name": "Mwaki", "artist": "Zerb"}]}}},
        "track.search:nothing at all": {"results": {"trackmatches": {"track": []}}},
        "artist.gettoptags": {"toptags": {"tag": [{"name": "Afro House"}]}},
    })  # fmt: skip
    return LastFm(client, "LFKEY")  # type: ignore[arg-type]


def test_parse_song() -> None:
    assert parse_song("Rampa - Crazy For It", None) == ("Rampa", "Crazy For It")
    assert parse_song("mwaki zerb", fake_lastfm()) == ("Zerb", "Mwaki")
    with pytest.raises(ValueError, match="Artist - Title"):
        parse_song("mwaki zerb", None)
    with pytest.raises(ValueError, match="found no song"):
        parse_song("nothing at all", fake_lastfm())


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    dp, _, _ = previews({"search:Zerb Mwaki": {"data": [hit(9, "Zerb", "Mwaki")]}, **DEEZER})
    bpm: GetSongBpm | None = None
    monkeypatch.setattr(song_module, "make_clients", lambda store, cfg: (fake_lastfm(), bpm, dp))


@pytest.mark.usefixtures("fake_clients")
def test_resolve_song() -> None:
    col = small_collection()
    owned = resolve_song("Adam Port - Move", col, None)
    assert owned.in_library and owned.track.id in {"1", "2"}
    new = resolve_song("mwaki zerb", col, None)
    t = new.track
    assert not new.in_library and t.id == EXTERNAL_ID
    assert (t.artist, t.title, t.bpm, t.camelot) == ("Zerb", "Mwaki", 121.5, "8A")
    assert t.genre == "afro house"  # from the artist's Last.fm tags
    assert t.key_confidence == DEFAULT_CONFIG.discovery.preview_key_confidence
    assert new.bpm_key_source == DEEZER_PREVIEW and not new.warnings
    unknown = resolve_song("Nobody - Nothing", col, None)
    assert unknown.track.bpm is None and "no BPM or key" in unknown.warnings[0]
    with pytest.raises(ValueError):
        resolve_song("   ", col, None)


def test_with_track_and_library_ids() -> None:
    col = small_collection()
    extra = Track(EXTERNAL_ID, "Mwaki", "Zerb")
    bigger = with_track(col, extra)
    assert EXTERNAL_ID in bigger.tracks and EXTERNAL_ID not in col.tracks
    assert library_ids(["1", EXTERNAL_ID, "3"]) == ("1", "3")


# ---------------------------------------------------------------- CLI and web


@pytest.mark.usefixtures("fake_clients")
def test_cli_suggest_song(fixture_5: Path) -> None:
    result = runner.invoke(cli.app, ["suggest", str(fixture_5), "--song", "mwaki zerb", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["seed"]["in_library"] is False and data["seed"]["bpm"] == 121.5
    suggested = [x["track"]["id"] for x in data["suggestions"]]
    assert suggested and EXTERNAL_ID not in suggested  # never the seed itself
    assert set(suggested) <= {"101", "102", "103", "104", "105"}
    both = runner.invoke(cli.app, ["suggest", str(fixture_5), "--song", "x - y", "--id", "101"])
    assert both.exit_code == 2
    table = runner.invoke(cli.app, ["suggest", str(fixture_5), "--song", "mwaki zerb"])
    assert "Not in your library" in table.output


@pytest.mark.usefixtures("fake_clients")
def test_cli_build_start_song(fixture_5: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    result = runner.invoke(
        cli.app,
        ["build", str(fixture_5), "--start-song", "mwaki zerb", "-n", "3", "-o", str(out),
         "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    ids = [p["track"]["id"] for p in data["positions"]]
    assert ids[0] == EXTERNAL_ID and len(ids) == 3
    assert any("not in your library" in w for w in data["warnings"])
    exported = load_collection(out)
    assert exported.playlists is not None
    playlist = next(n for _, n in exported.playlists.walk() if n.name == data["name"])
    assert list(playlist.track_keys) == ids[1:]  # the outside song can't be exported


@pytest.mark.usefixtures("fake_clients")
def test_web_song(fixture_5: Path, isolated_db: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import setsmith.web.app as web

    client = TestClient(web.create_app(fixture_5, isolated_db))
    data = client.get("/api/suggest", params={"song": "mwaki zerb"}).json()
    assert data["seed"]["in_library"] is False and data["seed"]["key"] == "8A"
    assert client.get("/api/suggest").status_code == 400  # neither track_id nor song
    owned = client.get("/api/suggest", params={"track_id": "101"}).json()
    assert owned["seed"]["in_library"] is True
    built = client.post("/api/build", json={"tracks": 3, "start_song": "mwaki zerb"}).json()
    assert built["set"]["positions"][0]["track"]["id"] == EXTERNAL_ID
    export = client.get(f"/api/sets/{built['id']}/export")
    assert export.status_code == 200 and EXTERNAL_ID.encode() not in export.content

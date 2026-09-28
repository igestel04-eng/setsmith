"""Discovery through Last.fm and GetSongBPM, with fake services (no network in tests)."""

import email.message
import io
import json
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import setsmith.cli as cli
from setsmith.discovery import keys, preview
from setsmith.discovery.discover import (
    DiscoveryUnavailable,
    discover,
    identity,
    make_sources,
    seed_query,
)
from setsmith.discovery.getsongbpm import GetSongBpm
from setsmith.discovery.http import JsonClient, ServiceError
from setsmith.discovery.lastfm import LastFm
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.store import Store
from setsmith.styles.profile import load_style

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SETSMITH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("SETSMITH_LASTFM_KEY", raising=False)
    monkeypatch.delenv("SETSMITH_GETSONGBPM_KEY", raising=False)
    # Deezer preview analysis is tested with fakes below; elsewhere it is off.
    monkeypatch.setattr(preview, "available", lambda: False)
    return tmp_path / "config"


class FakeClient:
    """Answers JsonClient.get from canned responses keyed by method or path."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, path: str, params: dict[str, str]) -> Any:
        self.calls.append((path, params))
        key = params.get("method") or path
        if key == "artist.gettoptracks":
            key = f"{key}:{params['artist']}"
        if key == "search/":
            key = f"search/:{params['lookup']}"
            return self.responses.get(key, {"search": {"error": "no result"}})
        return self.responses.get(key, {"error": 6, "message": "not found"})


LASTFM = {
    "track.getsimilar": {
        "similartracks": {
            "track": [
                {"name": "Lanterns", "artist": {"name": "Moana Reyes"}, "match": 0.9,
                 "url": "https://www.last.fm/music/Moana+Reyes/_/Lanterns"},
                {"name": "Sunwater (Extended Mix)", "artist": {"name": "Kaya Sol"}, "match": 0.95,
                 "url": "https://www.last.fm/music/Kaya+Sol/_/Sunwater"},  # seed itself
                {"name": "Night Tide", "artist": {"name": "Ines Okafor"}, "match": 0.8,
                 "url": "javascript:alert(1)"},  # hostile URL must not pass through
                {"name": "Glass Harbour", "artist": {"name": "Nils Ødegaard"}, "match": 0.7,
                 "url": ""},  # the library only has the Café Remix: a different recording
            ]
        }
    },
    # Also the scene: every candidate's artist is similar to the seed's artist.
    "artist.getsimilar": {"similarartists": {"artist": [
        {"name": "Nils Ødegaard", "match": "0.9"}, {"name": "Tomas Lind", "match": "0.6"},
        {"name": "Ines Okafor", "match": "0.5"},
    ]}},
    "artist.gettoptracks:Tomas Lind": {
        "toptracks": {"track": [{"name": "Velvet Drift", "url": "https://www.last.fm/music/Tomas+Lind/_/Velvet+Drift"}]}
    },
    "artist.gettoptags": {"toptags": {"tag": [{"name": "electronic"}, {"name": "Afro-House"}]}},
}  # fmt: skip

GETSONGBPM = {
    # Lanterns: the search hit lacks tempo, so the client follows up with /song/.
    "search/:lanterns": {
        "search": [{"id": "L1", "title": "Lanterns", "artist": {"name": "Moana Reyes"}}]
    },
    "song/": {"song": {"id": "L1", "title": "Lanterns", "tempo": "123", "key_of": "Em",
                       "open_key": "2m", "uri": "https://getsongbpm.com/song/lanterns/L1"}},
    "search/:night tide": {
        "search": [{"id": "N1", "title": "Night Tide", "artist": {"name": "Ines Okafor"},
                    "tempo": "140", "key_of": "C#m"}]
    },
    "search/:velvet drift": {
        "search": [{"id": "V1", "title": "Something Else", "artist": {"name": "Other"},
                    "tempo": "122"}]
    },
}  # fmt: skip


def sources() -> tuple[LastFm, GetSongBpm, FakeClient, FakeClient]:
    lf_client, bpm_client = FakeClient(LASTFM), FakeClient(GETSONGBPM)
    return (
        LastFm(lf_client, "LFKEY"),  # type: ignore[arg-type]
        GetSongBpm(bpm_client, "BPMKEY"),  # type: ignore[arg-type]
        lf_client,
        bpm_client,
    )


# ---------------------------------------------------------------- clients


def test_lastfm_parsing() -> None:
    lfm, *_ = sources()
    similar = lfm.similar_tracks("Kaya Sol", "Sunwater", 50)
    assert (similar[0].artist, similar[0].title) == ("Moana Reyes", "Lanterns")
    assert lfm.similar_artists("Kaya Sol", 10)[1] == ("Tomas Lind", 0.6)
    assert lfm.artist_tags("Anyone") == ["electronic", "Afro-House"]
    with pytest.raises(ServiceError, match="not found"):
        lfm.top_tracks("Unknown Artist", 5, 0.5)


def test_getsongbpm_lookup() -> None:
    _, bpm, _, client = sources()
    lanterns = bpm.lookup("Moana Reyes", "Lanterns (Original Mix)")
    assert lanterns is not None
    assert (lanterns.bpm, lanterns.camelot) == (123.0, "9A")  # open key 2m = E minor
    assert [p for p, _ in client.calls] == ["search/", "song/"]  # followed up for tempo
    night = bpm.lookup("Ines Okafor", "Night Tide")
    assert night is not None and (night.bpm, night.camelot) == (140.0, "12A")
    assert bpm.lookup("Tomas Lind", "Velvet Drift") is None  # search hit is a different song


# ---------------------------------------------------------------- pipeline


def test_discover_ranks_new_tracks(collection_5: Collection) -> None:
    lfm, bpm, _, _ = sources()
    seed = collection_5.tracks["101"]  # Kaya Sol - Sunwater (Extended Mix), 122 BPM, 8A
    result = discover(collection_5, seed, lfm, bpm, top=10)
    titles = [d.track.title for d in result.items]
    assert "Sunwater (Extended Mix)" not in titles  # the seed itself
    assert "Lanterns" not in titles  # already owned
    # Night Tide is a known clash (12A, 140 BPM) so it ranks below the unknowns; the
    # original "Glass Harbour" is not the owned Café Remix.
    assert titles == ["Glass Harbour", "Velvet Drift", "Night Tide"]
    night = result.items[2]
    assert night.track.genre == "afro house"  # from the artist's Last.fm tags
    assert "lastfm" not in night.links  # the javascript: URL was dropped
    assert night.links["soundcloud"].startswith("https://soundcloud.com/search/sounds?q=")
    assert result.items[1].track.bpm is None  # no reliable BPM match
    assert (result.candidates, result.in_library) == (4, 1)
    assert result.looked_up == 3 and result.without_tempo_key == 2
    # "no result" is not an error, but unknown BPM/key is called out
    assert len(result.warnings) == 1 and "2 of 3 results have no BPM/key data" in result.warnings[0]
    data = result.to_dict()
    assert {a["text"] for a in data["attribution"]} >= {"BPM and key data from GetSongBPM"}
    known = [(i["bpm_key_known"], i["rank_basis"]) for i in data["results"]]
    assert known == [
        (False, "Last.fm similarity"),
        (False, "Last.fm similarity"),
        (True, "transition score"),
    ]


def test_discover_lists_other_scenes_last(collection_5: Collection) -> None:
    # A crossover seed's "listeners also played" list pulls in mainstream hits; their
    # artist is not similar to the seed's artist, so they go last however high the match.
    responses = json.loads(json.dumps(LASTFM))
    responses["track.getsimilar"]["similartracks"]["track"].append(
        {"name": "Stadium Anthem", "artist": {"name": "Big Room Hero"}, "match": 1.0, "url": ""}
    )
    _, bpm, _, _ = sources()
    lfm = LastFm(FakeClient(responses), "LFKEY")  # type: ignore[arg-type]
    result = discover(collection_5, collection_5.tracks["101"], lfm, bpm, top=10)
    assert [d.track.title for d in result.items][-1] == "Stadium Anthem"
    last = result.to_dict()["results"][-1]
    assert (last["scene_match"], last["off_scene"], last["rank_basis"]) == (
        0.0,
        True,
        "different scene",
    )
    assert result.items[0].scene == pytest.approx(0.9)  # Glass Harbour by Nils Ødegaard
    assert any(
        "1 of 4 results are by artists outside Kaya Sol's scene" in w for w in result.warnings
    )


def test_discover_without_scene_data(collection_5: Collection) -> None:
    responses = {k: v for k, v in LASTFM.items() if k != "artist.getsimilar"}
    _, bpm, _, _ = sources()
    lfm = LastFm(FakeClient(responses), "LFKEY")  # type: ignore[arg-type]
    result = discover(collection_5, collection_5.tracks["101"], lfm, bpm, top=10)
    assert [d.track.title for d in result.items] == ["Glass Harbour", "Night Tide"]
    assert all(d.scene is None and not d.off_scene for d in result.items)
    assert sum("can't be checked against its scene" in w for w in result.warnings) == 1


def test_discover_estimates_from_deezer_previews(collection_5: Collection) -> None:
    # GetSongBPM has no match for Velvet Drift or Glass Harbour; their Deezer previews do.
    from setsmith.discovery.preview import DeezerPreviews

    class Deezer:
        def get(self, path: str, params: dict[str, str]) -> Any:
            titles = {"Tomas Lind Velvet Drift": "Velvet Drift"}
            title = titles.get(params.get("q", ""))
            if title is None:
                return {"data": []}
            item = {"id": 7, "title": title, "artist": {"name": "Tomas Lind"}}
            item["preview"] = "https://cdnt-preview.dzcdn.net/a.mp3"
            item["link"] = "https://www.deezer.com/track/7"
            return {"data": [item]}

    previews = DeezerPreviews(
        Deezer(),  # type: ignore[arg-type]
        analyze=lambda data, cfg: (122.0, "8A", 0.9, "fake"),
        fetch=lambda url, cfg: b"mp3",
    )
    lfm, bpm, _, _ = sources()
    seed = collection_5.tracks["101"]  # 122 BPM, 8A
    result = discover(collection_5, seed, lfm, bpm, top=10, previews=previews)
    velvet = next(d for d in result.items if d.track.title == "Velvet Drift")
    assert result.items[0] is velvet  # now a known good match
    assert velvet.bpm_key_source == "estimated from the Deezer preview"
    assert velvet.track.key_confidence == DEFAULT_CONFIG.discovery.preview_key_confidence
    assert velvet.links["deezer"] == "https://www.deezer.com/track/7"
    night = next(d for d in result.items if d.track.title == "Night Tide")
    assert night.bpm_key_source == "GetSongBPM"


def test_discover_known_good_match_ranks_first(collection_5: Collection) -> None:
    # Velvet Drift is the least similar on Last.fm, but once GetSongBPM knows it mixes
    # well with the seed (same BPM and key) it outranks the unscored candidates.
    lfm, _, _, _ = sources()
    responses = dict(GETSONGBPM)
    responses["search/:velvet drift"] = {
        "search": [{"id": "V2", "title": "Velvet Drift", "artist": {"name": "Tomas Lind"},
                    "tempo": "122", "key_of": "Am"}]
    }  # fmt: skip
    bpm = GetSongBpm(FakeClient(responses), "BPMKEY")  # type: ignore[arg-type]
    seed = collection_5.tracks["101"]
    result = discover(collection_5, seed, lfm, bpm, top=10)
    titles = [d.track.title for d in result.items]
    assert titles == ["Velvet Drift", "Glass Harbour", "Night Tide"]
    assert result.items[0].score.total >= DEFAULT_CONFIG.discovery.good_match_score
    assert "1 of 3 results have no BPM/key data" in result.warnings[0]


def test_discover_with_style(collection_5: Collection) -> None:
    lfm, bpm, _, _ = sources()
    result = discover(
        collection_5, collection_5.tracks["101"], lfm, bpm, style=load_style("keinemusik")
    )
    assert all(d.style_fit is not None for d in result.items)


def test_discover_without_getsongbpm(collection_5: Collection) -> None:
    lfm, *_ = sources()
    result = discover(collection_5, collection_5.tracks["101"], lfm, None)
    assert all(d.track.bpm is None for d in result.items)
    assert any("no GetSongBPM key" in w for w in result.warnings)


def test_bad_getsongbpm_key_fails_fast(collection_5: Collection) -> None:
    lfm, *_ = sources()
    failing = GetSongBpm(FakeClient({}), "BAD")  # type: ignore[arg-type]
    failing.client.responses = {}  # type: ignore[attr-defined]

    class Refusing:
        def get(self, path: str, params: dict[str, str]) -> Any:
            raise ServiceError("GetSongBPM answered HTTP 401: invalid key")

    failing.client = Refusing()  # type: ignore[assignment]
    with pytest.raises(ServiceError, match="401"):
        discover(collection_5, collection_5.tracks["101"], lfm, failing)


def test_seed_needs_artist_and_title(collection_5: Collection) -> None:
    lfm, bpm, _, _ = sources()
    with pytest.raises(ValueError, match="artist and a title"):
        discover(collection_5, collection_5.tracks["105"], lfm, bpm)  # untitled, no artist


# ---------------------------------------------------------------- keys


def test_keys_file_is_private_and_env_wins(
    isolated_keys: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert keys.get_key("lastfm") is None
    path = keys.set_key("lastfm", " abc123 ")
    assert keys.get_key("lastfm") == "abc123" and keys.key_source("lastfm") == "file"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.setenv("SETSMITH_LASTFM_KEY", "from-env")
    assert keys.get_key("lastfm") == "from-env" and keys.key_source("lastfm") == "environment"
    monkeypatch.delenv("SETSMITH_LASTFM_KEY")
    assert keys.remove_key("lastfm") and keys.get_key("lastfm") is None
    with pytest.raises(ValueError):
        keys.set_key("spotify", "x")


def test_make_sources_requires_lastfm() -> None:
    with pytest.raises(DiscoveryUnavailable, match=r"Last\.fm API key"):
        make_sources(None)
    keys.set_key("lastfm", "k")
    lfm, bpm = make_sources(None)
    assert lfm.api_key == "k" and bpm is None


# ---------------------------------------------------------------- HTTP client


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_http_client_caches_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen.append(request.full_url)
        return FakeResponse(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with Store(tmp_path / "s.db") as store:
        client = JsonClient("Svc", "https://api.example.test", min_interval_s=0, timeout_s=1,
                            cache=store, cache_ttl_s=60)  # fmt: skip
        assert client.get("x/", {"api_key": "SECRET", "q": "a"}) == {"ok": True}
        assert client.get("x/", {"api_key": "SECRET", "q": "a"}) == {"ok": True}
        assert len(seen) == 1 and "api_key=SECRET" in seen[0]  # second call from cache
        (key,) = [r[0] for r in store._conn.execute("SELECT key FROM http_cache")]
        assert "SECRET" not in key


def test_http_errors_hide_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            email.message.Message(),
            io.BytesIO(b"bad api_key=SECRET given"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    client = JsonClient("Svc", "https://api.example.test", min_interval_s=0, timeout_s=1)
    with pytest.raises(ServiceError) as info:
        client.get("x/", {"api_key": "SECRET"})
    assert "403" in str(info.value) and "SECRET" not in str(info.value)


# ---------------------------------------------------------------- CLI and web


def fake_make_sources(store: object, cfg: object = None) -> tuple[LastFm, GetSongBpm]:
    lfm, bpm, _, _ = sources()
    return lfm, bpm


def test_cli_discover(fixture_5: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = runner.invoke(cli.app, ["discover", str(fixture_5), "--id", "101"])
    assert missing.exit_code == 1 and "Last.fm API key" in missing.output

    monkeypatch.setattr(cli, "make_sources", fake_make_sources)
    result = runner.invoke(cli.app, ["discover", str(fixture_5), "--id", "101", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [r["title"] for r in data["results"]] == ["Glass Harbour", "Velvet Drift", "Night Tide"]
    table = runner.invoke(cli.app, ["discover", str(fixture_5), "-t", "Sunwater"])
    flat = " ".join(table.output.split())
    assert "Night" in flat and "GetSongBPM" in flat  # attribution shown


def test_cli_keys(isolated_keys: Path) -> None:
    assert "missing" in runner.invoke(cli.app, ["keys", "status"]).output
    assert runner.invoke(cli.app, ["keys", "set", "lastfm", "abc"]).exit_code == 0
    status = json.loads(runner.invoke(cli.app, ["keys", "status", "--json"]).output)
    assert status == {"lastfm": "file", "getsongbpm": None}
    assert "abc" not in runner.invoke(cli.app, ["keys", "status"]).output  # never printed
    assert runner.invoke(cli.app, ["keys", "set", "spotify", "x"]).exit_code == 2
    prompted = runner.invoke(cli.app, ["keys", "set", "getsongbpm"], input="hiddenkey42\n")
    assert prompted.exit_code == 0 and "hiddenkey42" not in prompted.output
    assert keys.get_key("getsongbpm") == "hiddenkey42"
    placeholder = runner.invoke(cli.app, ["keys", "set", "lastfm", "PASTE_YOUR_API_KEY"])
    assert placeholder.exit_code == 2 and "placeholder" in placeholder.output
    odd = runner.invoke(cli.app, ["keys", "set", "lastfm", "not-a-hex-key"])
    assert odd.exit_code == 0 and "32 characters" in odd.output  # saved, with a warning


def test_web_discover(fixture_5: Path, isolated_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import setsmith.web.app as web

    client = TestClient(web.create_app(fixture_5, isolated_db))
    assert client.get("/api/info").json()["discovery"] == {"lastfm": False, "getsongbpm": False}
    assert client.get("/api/discover", params={"track_id": "101"}).status_code == 400
    monkeypatch.setattr(web, "make_sources", fake_make_sources)
    data = client.get("/api/discover", params={"track_id": "101", "style": "keinemusik"}).json()
    assert [r["title"] for r in data["results"]] == ["Glass Harbour", "Velvet Drift", "Night Tide"]
    assert client.get("/api/discover", params={"track_id": "nope"}).status_code == 404


# ---------------------------------------------------------------- SoundCloud-style metadata


@pytest.mark.parametrize(
    ("artist", "title", "query", "artists", "clean"),
    [
        ("Adam Port", "Adam Port, Stryv, Malachiii - Move (Extended)",
         ("Adam Port", "Move"), {"adam port", "stryv", "malachiii"}, "move"),
        ("Keinemusik", "Keinemusik (Rampa, &ME, Adam Port) - Say What (feat. Chuala)",
         ("Keinemusik", "Say What"), {"keinemusik", "rampa", "me", "adam port"}, "say what"),
        ("Rampa", "Rampa, Sparrow & Barbossa - Champion (VOD)",
         ("Rampa", "Champion"), {"rampa", "sparrow", "barbossa"}, "champion"),
        ("Imad", "Rufus Du Sol - Innerbloom (Imad & Dennis Louvra Remix)",
         ("Rufus Du Sol", "Innerbloom"), {"imad", "rufus du sol"},
         "innerbloom imad & dennis louvra remix"),  # the remix is a different recording
        ("Keinemusik", "Keinemusik (&ME, Rampa, Adam Port) - Discoteca feat. Sofie",
         ("Keinemusik", "Discoteca"), {"keinemusik", "rampa", "me", "adam port"}, "discoteca"),
        ("Kaya Sol", "Sunwater (Extended Mix)", ("Kaya Sol", "Sunwater"), {"kaya sol"}, "sunwater"),
    ],
)  # fmt: skip
def test_soundcloud_style_titles(
    artist: str, title: str, query: tuple[str, str], artists: set[str], clean: str
) -> None:
    assert seed_query(Track(id="x", artist=artist, title=title)) == query
    assert identity(artist, title) == (frozenset(artists), clean)


def test_owned_soundcloud_upload_is_not_rediscovered() -> None:
    col = Collection(
        tracks={"1": Track(id="1", artist="Adam Port", title="Adam Port, Stryv - Move (Extended)")}
    )
    lf_client = FakeClient({
        "track.getsimilar": {"similartracks": {"track": [
            {"name": "Move", "artist": {"name": "Stryv"}, "match": 1.0, "url": ""},
            {"name": "Positions", "artist": {"name": "Stryv"}, "match": 0.9, "url": ""},
        ]}},
    })  # fmt: skip
    result = discover(col, col.tracks["1"], LastFm(lf_client, "k"), None)  # type: ignore[arg-type]
    assert [d.track.title for d in result.items] == ["Positions"]
    assert result.in_library == 1
    assert lf_client.calls[0][1]["track"] == "Move"  # queried with the unpacked title


def test_library_matching_real_world_variants() -> None:
    col = Collection(
        tracks={
            "1": Track(id="1", artist="Keinemusik",
                       title="Keinemusik (Rampa, &ME, Adam Port) - Muyè"),
            "2": Track(id="2", artist="MoBlack Records", title="Yamore (feat. Cesária Évora)"),
        }
    )  # fmt: skip
    lf_client = FakeClient({
        "track.getsimilar": {"similartracks": {"track": [
            {"name": "Muyè", "artist": {"name": "Rampa"}, "match": 1.0, "url": ""},
            {"name": "Yamore - Edit Version", "artist": {"name": "MoBlack"}, "match": 0.9,
             "url": ""},
            {"name": "Yamore - Francis Mercier Remix", "artist": {"name": "MoBlack"}, "match": 0.8,
             "url": ""},
            {"name": "Ngeke", "artist": {"name": "MoBlack"}, "match": 0.7, "url": ""},
            {"name": "Mayana", "artist": {"name": "MoBlack"}, "match": 0.6, "url": ""},
        ]}},
    })  # fmt: skip
    seed = Track(id="0", artist="Adam Port", title="Move")
    col.tracks["0"] = seed
    result = discover(col, seed, LastFm(lf_client, "k"), None)  # type: ignore[arg-type]
    titles = [d.track.title for d in result.items]
    assert "Muyè" not in titles  # owned via the Keinemusik member list
    assert "Yamore - Edit Version" not in titles  # owned: an edit of the same song
    assert "Yamore - Francis Mercier Remix" in titles  # a remix is a different recording
    assert len(titles) == 2  # at most two per artist


@pytest.mark.parametrize(
    ("artist", "title", "artists", "clean"),
    [
        ("MoBlack Records", "Yamore (feat. Cesária Évora, Benja (NL) & Franc Fala)",
         {"moblack records"}, "yamore"),
        ("MoBlack\uff0c Salif Keïta", "Yamore (feat. Cesária Évora, Benja (NL) & Franc Fala)",
         {"moblack", "salif keita"}, "yamore"),  # full-width comma
        ("A", "Song (feat. B) (C Remix)", {"a"}, "song c remix"),  # the remix survives
    ],
)  # fmt: skip
def test_nested_feat_and_fullwidth_separators(
    artist: str, title: str, artists: set[str], clean: str
) -> None:
    assert identity(artist, title) == (frozenset(artists), clean)


def test_cli_keys_from_clipboard(isolated_keys: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import subprocess

    class Done:
        def __init__(self, out: str) -> None:
            self.stdout = out

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pbpaste")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done("  clipkey123\n"))
    result = runner.invoke(cli.app, ["keys", "set", "getsongbpm", "--from-clipboard"])
    assert result.exit_code == 0 and "clipkey123" not in result.output
    assert keys.get_key("getsongbpm") == "clipkey123"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done("two words"))
    bad = runner.invoke(cli.app, ["keys", "set", "getsongbpm", "--from-clipboard"])
    assert bad.exit_code == 2 and "clipboard" in bad.output
    assert keys.get_key("getsongbpm") == "clipkey123"  # unchanged


def test_getsongbpm_band_name_variants() -> None:
    client = FakeClient({"search/:innerbloom": {"search": [
        {"id": "I1", "title": "Innerbloom", "artist": {"name": "RÜFÜS"}, "tempo": "122",
         "open_key": "11m"},
        {"id": "I2", "title": "Innerbloom", "artist": {"name": "THE SOUND BEE HD"}, "tempo": "138"},
    ]}, "search/:yamore": {"search": [
        {"id": "Y1", "title": "Yamore", "artist": {"name": "Salif Keita"}, "tempo": "160"},
    ]}})  # fmt: skip
    bpm = GetSongBpm(client, "k")  # type: ignore[arg-type]
    innerbloom = bpm.lookup("RÜFÜS DU SOL", "Innerbloom")
    assert innerbloom is not None and innerbloom.bpm == 122.0  # older band name accepted
    assert bpm.lookup("MoBlack", "Yamore") is None  # a different artist's recording
    assert client.calls[0][1] == {"api_key": "k", "type": "song", "lookup": "innerbloom"}

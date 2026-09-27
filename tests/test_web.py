"""Phase 6 web API. Skipped without the web extra."""

import hashlib
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from setsmith.analysis.liveset import LiveSet, MatchRecord
from setsmith.io.rekordbox_xml import load_collection
from setsmith.store import Store
from setsmith.web.app import create_app


@pytest.fixture
def client(fixture_pool: Path, isolated_db: Path) -> TestClient:
    return TestClient(create_app(fixture_pool, isolated_db))


def test_index_and_static(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "<title>Setsmith</title>" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_info(client: TestClient) -> None:
    info = client.get("/api/info").json()
    assert info["tracks"] == 300
    assert info["curves"] == ["warm_up", "peak_time", "closing", "journey"]
    assert {s["key"] for s in info["styles"]} == {"brunello", "franky_rizardo", "keinemusik"}
    assert info["learned_available"] is False


def test_track_search(client: TestClient, fixture_pool: Path) -> None:
    col = load_collection(fixture_pool)
    some = next(t for t in col.tracks.values() if t.artist)
    words = some.artist.split()[0].lower()
    hits = client.get("/api/tracks", params={"q": words, "limit": 5}).json()
    assert 0 < len(hits) <= 5
    assert all(words in f"{h['artist']} {h['title']}".lower() for h in hits)
    assert client.get("/api/tracks", params={"q": some.id}).json()[0]["id"] == some.id
    assert client.get("/api/tracks", params={"q": "  "}).json() == []


def test_suggest(client: TestClient, fixture_pool: Path) -> None:
    seed = next(t for t in load_collection(fixture_pool).tracks.values() if t.bpm)
    data = client.get("/api/suggest", params={"track_id": seed.id, "top": 5}).json()
    assert data["seed"]["id"] == seed.id and len(data["suggestions"]) == 5
    first = data["suggestions"][0]
    assert first["explain"] and first["style_fit"] is None
    styled = client.get("/api/suggest", params={"track_id": seed.id, "style": "keinemusik"}).json()
    assert styled["style"] == "Inspired by Keinemusik"
    assert styled["suggestions"][0]["style_fit"] is not None
    assert client.get("/api/suggest", params={"track_id": "nope"}).status_code == 404


@pytest.mark.parametrize(
    "style", ["nope", "../setsmith/styles/profiles/keinemusik.json", "/etc/passwd"]
)
def test_styles_by_name_only(client: TestClient, fixture_pool: Path, style: str) -> None:
    seed = next(iter(load_collection(fixture_pool).tracks))
    result = client.get("/api/suggest", params={"track_id": seed, "style": style})
    assert result.status_code == 400
    assert client.post("/api/build", json={"tracks": 4, "style": style}).status_code == 400


def test_learned_needs_data(client: TestClient) -> None:
    assert client.post("/api/build", json={"tracks": 4, "learned": True}).status_code == 400


def test_build_export_and_report(
    client: TestClient, fixture_pool: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = hashlib.sha256(fixture_pool.read_bytes()).hexdigest()
    built = client.post(
        "/api/build", json={"tracks": 8, "style": "franky_rizardo", "name": "Web test"}
    )
    assert built.status_code == 200, built.text
    data = built.json()
    positions = data["set"]["positions"]
    assert len(positions) == 8 and data["set"]["name"] == "Web test"
    assert all(125 <= p["track"]["bpm"] <= 130 for p in positions)
    assert len(data["timeline"]) == 8
    starts = [t["start_s"] for t in data["timeline"]]
    assert starts == sorted(starts)
    assert all(t["end_s"] > t["start_s"] for t in data["timeline"])
    assert positions[0]["explain"] and positions[-1]["explain"] == []
    assert len(data["curve_targets"]) == 8

    made: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(dir=tmp_path, prefix="export-")
        made.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
    export = client.get(f"/api/sets/{data['id']}/export")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/xml")
    assert "Web%20test.xml" in export.headers["content-disposition"]  # RFC 5987 encoding
    out = tmp_path / "downloaded.xml"
    out.write_bytes(export.content)
    exported = load_collection(out)
    assert exported.playlists is not None
    names = [n.name for _, n in exported.playlists.walk()]
    assert names[:3] == ["ROOT", "Setsmith", "Web test"]
    assert made and not made[0].exists()  # the temporary export folder is removed
    assert hashlib.sha256(fixture_pool.read_bytes()).hexdigest() == before

    report = client.get(f"/api/sets/{data['id']}/report")
    assert report.status_code == 200 and report.text.startswith("# Web test")
    assert client.get("/api/sets/unknown/export").status_code == 404


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"tracks": 4, "curve": "sunrise"}, 400),
        ({"tracks": 4, "bpm_min": 300}, 422),
        ({"tracks": 1}, 422),  # validation: at least 2 tracks
        ({"minutes": -5}, 422),
    ],
)
def test_build_errors(client: TestClient, body: dict[str, object], status: int) -> None:
    assert client.post("/api/build", json=body).status_code == status


def test_livesets(client: TestClient, isolated_db: Path) -> None:
    assert client.get("/api/livesets").json() == []
    record = MatchRecord(
        position=1, raw="01. A - B", artist="A", title="B", start_s=None, unknown=False,
        together=False, track_id="1", track_path="/m/a.mp3", track_display="A - B", score=1.0,
        bpm=124.0, camelot="8A",
    )  # fmt: skip
    with Store(isolated_db) as store:
        liveset_id = store.save_liveset(
            LiveSet("Friday", "friday.txt", None, None, [record]).to_dict()
        )
    listed = client.get("/api/livesets").json()
    assert [(x["id"], x["name"]) for x in listed] == [(liveset_id, "Friday")]
    detail = client.get(f"/api/livesets/{liveset_id}").json()
    assert detail["liveset"]["name"] == "Friday" and detail["stats"]["matched"] == 1
    assert client.get("/api/livesets/999").status_code == 404


def test_rejects_foreign_host_headers(client: TestClient) -> None:
    # Guards against DNS rebinding: only requests addressed to a local name are served.
    assert client.get("/api/info", headers={"host": "evil.example"}).status_code == 400
    assert client.get("/api/info", headers={"host": "localhost:8765"}).status_code == 200
    assert client.get("/api/info", headers={"host": "127.0.0.1:8765"}).status_code == 200

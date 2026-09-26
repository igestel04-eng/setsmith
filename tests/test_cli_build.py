import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from setsmith.cli import app
from setsmith.io.rekordbox_xml import load_collection

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "setsmith.db"


def test_build_help() -> None:
    for args in (["build", "--help"], ["feedback", "--help"], ["feedback", "log", "--help"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output


def test_build_table_export_and_report(fixture_pool: Path, tmp_path: Path, db: Path) -> None:
    out, report = tmp_path / "set.xml", tmp_path / "set.md"
    args = ["build", str(fixture_pool), "--minutes", "60", "--curve", "warm_up", "--name", "Test"]
    result = runner.invoke(
        app, [*args, "--out", str(out), "--report", str(report), "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "Test" in result.output and "Alternates" in result.output
    assert "Import Playlist" in " ".join(result.output.split())

    exported = load_collection(out)
    assert exported.playlists is not None
    names = [n.name for _, n in exported.playlists.walk()]
    assert names == ["ROOT", "Setsmith", "Test", "Test (alternates)"]
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Test") and "## Transitions" in text and "## Alternates" in text


def test_build_json_and_cache(fixture_pool: Path, db: Path) -> None:
    args = ["build", str(fixture_pool), "--tracks", "10", "--db", str(db), "--json"]
    first = json.loads(runner.invoke(app, args).output)
    second = json.loads(runner.invoke(app, args).output)
    assert len(first["positions"]) == 10
    assert first["stats"]["pairs_scored"] > 0 and first["stats"]["pairs_from_cache"] == 0
    assert second["stats"]["pairs_scored"] == 0 and second["stats"]["pairs_from_cache"] > 0
    assert [p["track"]["id"] for p in first["positions"]] == [
        p["track"]["id"] for p in second["positions"]
    ]
    first_step = first["positions"][0]["transition_to_next"]
    assert set(first_step["components"]) == {"harmonic", "tempo", "energy", "genre_style", "extras"}
    assert first["export"] is None


def test_build_options(fixture_pool: Path, db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "build", str(fixture_pool), "--tracks", "8", "--curve", "0:4,1:7",
            "--bpm-min", "120", "--bpm-max", "125", "--genre", "house", "--genre", "tech-house",
            "--exclude", "1", "--no-cache", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["curve"] == "custom"
    for p in data["positions"]:
        assert 120 <= p["track"]["bpm"] <= 125
        assert p["track"]["genre"] in ("House", "Tech House")
        assert p["track"]["id"] != "1"


def test_build_write_comments(fixture_pool: Path, tmp_path: Path, db: Path) -> None:
    out = tmp_path / "set.xml"
    args = ["build", str(fixture_pool), "--tracks", "5", "--out", str(out), "--db", str(db)]
    assert runner.invoke(app, [*args, "--name", "N", "--write-comments"]).exit_code == 0
    comments = [t.comments for t in load_collection(out).tracks.values()]
    assert sum(c.startswith("[Setsmith N #") for c in comments) == 5


def test_build_errors(fixture_pool: Path, fixture_5: Path, db: Path) -> None:
    bad_curve = runner.invoke(app, ["build", str(fixture_pool), "--curve", "sunrise"])
    assert bad_curve.exit_code == 2
    empty = runner.invoke(app, ["build", str(fixture_5), "--bpm-min", "300", "--db", str(db)])
    assert empty.exit_code == 3
    overwrite = runner.invoke(
        app, ["build", str(fixture_5), "--tracks", "3", "--out", str(fixture_5), "--db", str(db)]
    )
    assert overwrite.exit_code == 3 and "input" in overwrite.output


def test_feedback_log_and_list(fixture_5: Path, db: Path) -> None:
    result = runner.invoke(
        app,
        [
            "feedback", "log", str(fixture_5), "good", "--from", "Kaya Sol - Sunwater",
            "--to-id", "102", "--note", "chant lines up", "--db", str(db),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Logged #1" in result.output
    bad = runner.invoke(
        app,
        ["feedback", "log", str(fixture_5), "bad", "--from-id", "101", "--to-id", "104",
         "--type", "cut", "--bars", "8", "--db", str(db)],
    )  # fmt: skip
    assert bad.exit_code == 0, bad.output

    listing = json.loads(runner.invoke(app, ["feedback", "list", "--db", str(db), "--json"]).output)
    assert [e["verdict"] for e in listing] == ["good", "bad"]
    assert listing[0]["transition_type"] == "long_blend" and listing[0]["length_bars"] == 32
    assert listing[1]["transition_type"] == "cut" and listing[1]["length_bars"] == 8
    table = runner.invoke(app, ["feedback", "list", "--db", str(db)])
    assert "2 logged: 1 good, 1 bad" in table.output


def test_feedback_bad_verdict(fixture_5: Path, db: Path) -> None:
    result = runner.invoke(
        app, ["feedback", "log", str(fixture_5), "meh", "--from-id", "101", "--to-id", "102"]
    )
    assert result.exit_code == 2

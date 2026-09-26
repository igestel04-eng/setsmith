import json
from pathlib import Path

from typer.testing import CliRunner

from setsmith.cli import app

runner = CliRunner()


def test_help_for_every_command() -> None:
    for args in (["--help"], ["suggest", "--help"], ["info", "--help"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "--help" in result.output
    assert "--json" in runner.invoke(app, ["suggest", "--help"]).output
    assert "--json" in runner.invoke(app, ["info", "--help"]).output


def test_suggest_table(fixture_5: Path) -> None:
    result = runner.invoke(app, ["suggest", str(fixture_5), "--track", "Kaya Sol - Sunwater"])
    assert result.exit_code == 0, result.output
    assert "Moana Reyes - Lanterns" in result.output
    assert "long_blend" in result.output


def test_suggest_explain(fixture_5: Path) -> None:
    result = runner.invoke(app, ["suggest", str(fixture_5), "-t", "Sunwater", "--explain"])
    assert result.exit_code == 0, result.output
    assert "8A -> 9A (adjacent, +1)" in result.output


def test_suggest_json(fixture_5: Path) -> None:
    result = runner.invoke(app, ["suggest", str(fixture_5), "--id", "101", "--top", "2", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["seed"]["id"] == "101"
    assert [s["track"]["id"] for s in data["suggestions"]] == ["102", "103"]
    first = data["suggestions"][0]["score"]
    assert first["suggested_type"] == "long_blend"
    assert first["suggested_length_bars"] == 32
    assert set(first["components"]) == {"harmonic", "tempo", "energy", "genre_style", "extras"}


def test_suggest_negative_energy_delta(fixture_5: Path) -> None:
    result = runner.invoke(
        app, ["suggest", str(fixture_5), "--id", "101", "--energy-delta", "-1", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["energy_target"] == -1


def test_unknown_track(fixture_5: Path) -> None:
    result = runner.invoke(app, ["suggest", str(fixture_5), "-t", "Nobody - Nothing"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["suggest", str(fixture_5), "--id", "999", "--json"])
    assert result.exit_code == 2
    assert "999" in json.loads(result.output)["error"]


def test_ambiguous_track_lists_candidates(fixture_50: Path) -> None:
    # Tracks 1 and 5 are the same song with different TrackIDs.
    result = runner.invoke(
        app, ["suggest", str(fixture_50), "-t", "Kaya Sol - Tidal Drum", "--json"]
    )
    assert result.exit_code == 2
    ids = {c["id"] for c in json.loads(result.output)["candidates"]}
    assert {"1", "5"} <= ids


def test_missing_seed_option(fixture_5: Path) -> None:
    result = runner.invoke(app, ["suggest", str(fixture_5)])
    assert result.exit_code == 2


def test_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("not xml", encoding="utf-8")
    result = runner.invoke(app, ["info", str(bad)])
    assert result.exit_code == 1


def test_info(fixture_5: Path) -> None:
    result = runner.invoke(app, ["info", str(fixture_5), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["tracks"] == 5
    assert data["playlists"] == 2
    assert data["coverage"]["key"] == 4
    assert runner.invoke(app, ["info", str(fixture_5)]).exit_code == 0


def test_input_file_untouched(fixture_5: Path) -> None:
    before = fixture_5.read_bytes()
    runner.invoke(app, ["suggest", str(fixture_5), "--id", "101"])
    assert fixture_5.read_bytes() == before

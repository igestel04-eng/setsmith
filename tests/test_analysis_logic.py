"""Phase 3 logic that does not need the audio libraries."""

import os
from pathlib import Path
from typing import Any

import pytest

from setsmith.analysis.apply import apply_analyses, calibrate_energy, load_and_apply
from setsmith.analysis.audio import (
    TrackAnalysis,
    bar_starts,
    detect_sections,
    key_from_chroma,
    location_to_path,
    structure_from_segments,
)
from setsmith.keys.camelot import all_keys
from setsmith.model.collection import Collection
from setsmith.model.track import EnergySource, TempoMarker, Track
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.store import Store

# ---------------------------------------------------------------- locations


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("file://localhost/Users/dj/Music/A%20B%20(Mix).mp3", "/Users/dj/Music/A B (Mix).mp3"),
        ("file://localhost/Users/dj/%C3%98degaard.aiff", "/Users/dj/Ødegaard.aiff"),
        ("file:///Users/dj/a.wav", "/Users/dj/a.wav"),
        ("file://localhost/C:/Music/a.mp3", "C:/Music/a.mp3"),
        ("/Users/dj/plain.mp3", "/Users/dj/plain.mp3"),
    ],
)
def test_location_to_path(location: str, expected: str) -> None:
    assert location_to_path(location) == Path(expected)


@pytest.mark.parametrize("location", ["", "https://example.com/a.mp3", "file://nas/share/a.mp3"])
def test_non_local_locations(location: str) -> None:
    assert location_to_path(location) is None


# ---------------------------------------------------------------- beat grid


def test_bar_starts_simple_grid() -> None:
    bars = bar_starts([TempoMarker(0.0, 120.0, "4/4", 1)], 10.0)
    assert bars == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])


def test_bar_starts_before_first_marker_and_mid_bar() -> None:
    # Marker on beat 3 at 3.0s (120 BPM): downbeats at 4.0, 6.0, ... and 2.0, 0.0 before.
    bars = bar_starts([TempoMarker(3.0, 120.0, "4/4", 3)], 9.0)
    assert bars == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])


def test_bar_starts_tempo_change_and_three_four() -> None:
    markers = [TempoMarker(0.0, 120.0, "4/4", 1), TempoMarker(4.0, 60.0, "3/4", 1)]
    assert bar_starts(markers, 13.0) == pytest.approx([0.0, 2.0, 4.0, 7.0, 10.0])
    assert bar_starts([], 10.0) == []


# ---------------------------------------------------------------- sections


def levels(
    intro: int, main: int, outro: int, low: float = -30.0, high: float = -10.0
) -> list[float]:
    return [low] * intro + [high] * main + [low] * outro


@pytest.mark.parametrize(
    ("lv", "expected"),
    [
        (levels(8, 24, 16), (8, 16)),
        (levels(16, 32, 32), (16, 32)),
        (levels(0, 64, 0), (0, 0)),
        (levels(7, 40, 9), (8, 8)),  # snapped to 8-bar phrases
        (levels(40, 16, 8), (None, 8)),  # a 40-bar "intro" in a 64-bar track is implausible
        ([-10.0] * 5, (None, None)),  # too short to judge
    ],
)
def test_detect_sections(lv: list[float], expected: tuple[int | None, int | None]) -> None:
    assert detect_sections(lv) == expected


def test_detect_sections_ignores_short_breakdown() -> None:
    lv = levels(16, 16, 0) + [-30.0] * 8 + [-10.0] * 16 + [-30.0] * 16
    assert detect_sections(lv) == (16, 16)


def test_structure_from_segments() -> None:
    downbeats = [i * 2.0 for i in range(40)]  # 40 bars
    segments = [
        (0.0, 0.1, "start"),
        (0.1, 32.0, "intro"),
        (32.0, 64.0, "chorus"),
        (64.0, 80.0, "outro"),
        (80.0, 81.0, "end"),
    ]
    assert structure_from_segments(downbeats, segments) == (16, 8)
    no_labels = [(0.0, 80.0, "verse")]
    assert structure_from_segments(downbeats, no_labels) == (0, 0)
    assert structure_from_segments([], segments) == (None, None)


# ---------------------------------------------------------------- key templates


@pytest.mark.parametrize("key", all_keys(), ids=str)
def test_key_from_chroma_recovers_every_key(key: object) -> None:
    from setsmith.keys.camelot import CamelotKey

    assert isinstance(key, CamelotKey)
    tonic = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"].index(
        key.name.split()[0].replace("Db", "C#")
    )
    profile = DEFAULT_CONFIG.analysis.kk_minor if key.is_minor else DEFAULT_CONFIG.analysis.kk_major
    chroma = [profile[(pc - tonic) % 12] for pc in range(12)]
    detected, corr = key_from_chroma(chroma)
    assert detected == str(key)
    assert corr == pytest.approx(1.0)


def test_key_from_chroma_silence() -> None:
    assert key_from_chroma([0.0] * 12) == (None, 0.0)


# ---------------------------------------------------------------- energy calibration


def analysis(path: str, loud: float, **kw: object) -> TrackAnalysis:
    base: dict[str, object] = {
        "path": path, "mtime": 1.0, "size": 10, "version": DEFAULT_CONFIG.analysis.version,
        "analyzed_at": "2026-01-01T00:00:00+00:00", "duration_s": 300.0, "loudness_db": loud,
        "onset_rate": 3.0 + loud / 10, "spectral_centroid_hz": 1500.0 + loud * 10,
        "spectral_flux": 1.0 + loud / 20,
    }  # fmt: skip
    base.update(kw)
    return TrackAnalysis(**base)  # type: ignore[arg-type]


def test_calibrate_energy_spreads_over_scale() -> None:
    analyses = [analysis(str(i), loud) for i, loud in enumerate([-20.0, -8.0, -14.0, -11.0])]
    assert calibrate_energy(analyses) == [1.0, 10.0, 4.0, 7.0]


def test_calibrate_energy_edge_cases() -> None:
    assert calibrate_energy([]) == []
    assert calibrate_energy([analysis("a", -10.0)]) == [5.5]
    same = [analysis("a", -10.0), analysis("b", -10.0), analysis("c", -5.0)]
    energies = calibrate_energy(same)
    assert energies[0] == energies[1] < energies[2]


# ---------------------------------------------------------------- applying


def collection_of(*tracks: Track) -> Collection:
    return Collection(tracks={t.id: t for t in tracks})


def track(tid: str, key: str | None, **kw: Any) -> Track:
    return Track(
        id=tid, camelot=key, key_raw=key, location=f"file://localhost/music/{tid}.mp3", **kw
    )


def test_key_agreement_sets_confidence() -> None:
    col = collection_of(
        track("agree", "8A"),
        track("related", "8A"),
        track("clash", "8A"),
        track("untagged", None),
        track("undetected", "8A"),
    )
    detected = {"agree": "8A", "related": "8B", "clash": "2A", "untagged": "5A", "undetected": None}
    analyses = {
        f"/music/{tid}.mp3": analysis(f"/music/{tid}.mp3", -10.0, key_camelot=key)
        for tid, key in detected.items()
    }
    cov = apply_analyses(col, analyses)
    t = col.tracks
    assert t["agree"].key_confidence == 1.0
    assert t["related"].key_confidence == 0.7
    assert t["clash"].key_confidence == 0.4 and t["clash"].camelot == "8A"  # tag kept
    assert t["untagged"].camelot == "5A" and t["untagged"].key_confidence == 0.8
    assert t["undetected"].key_confidence == 1.0 and t["undetected"].detected_camelot is None
    assert (cov.key_agree, cov.key_related, cov.key_disagree, cov.key_from_audio) == (1, 1, 1, 1)
    assert cov.analyzed == 5


def test_energy_tag_wins_and_structure_applies() -> None:
    col = collection_of(
        track("tagged", "8A", energy=7.0, energy_source=EnergySource.COMMENT_TAG),
        track("rated", "8A", energy=6.0, energy_source=EnergySource.RATING),
    )
    analyses = {
        "/music/tagged.mp3": analysis("/music/tagged.mp3", -5.0, intro_bars=16, outro_bars=32),
        "/music/rated.mp3": analysis("/music/rated.mp3", -20.0),
    }
    cov = apply_analyses(col, analyses)
    tagged, rated = col.tracks["tagged"], col.tracks["rated"]
    assert tagged.energy == 7.0 and tagged.energy_source == EnergySource.COMMENT_TAG
    assert rated.energy == 1.0 and rated.energy_source == EnergySource.AUDIO
    assert (tagged.intro_bars, tagged.outro_bars) == (16, 32)
    assert rated.intro_bars is None
    assert cov.energy_from_audio == 1 and cov.structure_known == 1


# ---------------------------------------------------------------- store + stale files


def test_store_analyses(tmp_path: Path) -> None:
    with Store(tmp_path / "s.db") as store:
        store.save_analysis(analysis("/a.mp3", -10.0, mtime=1.0))
        store.save_analysis(analysis("/a.mp3", -9.0, mtime=2.0))  # replaces the old result
        assert store.count_analyses() == 1
        assert store.load_analyses([("/a.mp3", 1.0, 10)], DEFAULT_CONFIG.analysis.version) == {}
        found = store.load_analyses([("/a.mp3", 2.0, 10)], DEFAULT_CONFIG.analysis.version)
        assert found["/a.mp3"].loudness_db == -9.0
        assert store.load_analyses([("/a.mp3", 2.0, 10)], version=999) == {}
        assert store.analyzed_paths() == {"/a.mp3"}


def test_load_and_apply_detects_stale_and_missing(tmp_path: Path) -> None:
    fresh, changed = tmp_path / "fresh.mp3", tmp_path / "changed.mp3"
    for f in (fresh, changed):
        f.write_bytes(b"x" * 10)
    col = collection_of(
        Track(id="1", location=f"file://localhost{fresh}"),
        Track(id="2", location=f"file://localhost{changed}"),
        Track(id="3", location=f"file://localhost{tmp_path}/gone.mp3"),
    )
    with Store(tmp_path / "s.db") as store:
        for f in (fresh, changed):
            st = f.stat()
            store.save_analysis(analysis(str(f), -10.0, mtime=st.st_mtime, size=st.st_size))
        os.utime(changed, (1.0, 1.0))
        cov = load_and_apply(col, store)
    assert (cov.analyzed, cov.stale, cov.missing_file, cov.not_analyzed) == (1, 1, 1, 0)
    assert col.tracks["1"].energy_source == EnergySource.AUDIO
    assert col.tracks["2"].energy_source is None


def test_read_only_commands_do_not_create_a_database(fixture_5: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from setsmith.cli import app

    db = tmp_path / "nothing-here.db"
    runner = CliRunner()
    for args in (["suggest", str(fixture_5), "--id", "101"], ["info", str(fixture_5)]):
        assert runner.invoke(app, [*args, "--db", str(db)]).exit_code == 0
    assert not db.exists()

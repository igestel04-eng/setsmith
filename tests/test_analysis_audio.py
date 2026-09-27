"""Phase 3 analysis on synthesized audio. Skipped without the audio extra."""

import json
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("librosa")

from audio_synth import render_track
from synth import SynthTrack, write_collection
from typer.testing import CliRunner

from setsmith.analysis.audio import (
    AnalysisError,
    AnalysisJob,
    AnalysisOptions,
    analyze_file,
    analyze_many,
    has_module,
)
from setsmith.cli import app
from setsmith.model.track import TempoMarker

pytestmark = pytest.mark.audio

GRID = (TempoMarker(0.0, 124.0, "4/4", 1),)
BACKENDS = [
    pytest.param(True, id="essentia", marks=pytest.mark.skipif(
        not has_module("essentia"), reason="Essentia not installed")),
    pytest.param(False, id="librosa"),
]  # fmt: skip


@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    render_track(d / "busy.wav")  # intro 8, main 24, outro 16
    render_track(d / "sparse.wav", intro_bars=16, outro_bars=8, busy=False, gain=0.3, seed=1)
    return d


@pytest.mark.parametrize("essentia", BACKENDS)
def test_key_and_structure_with_grid(audio_dir: Path, essentia: bool) -> None:
    job = AnalysisJob("1", str(audio_dir / "busy.wav"), GRID, 124.0)
    result = analyze_file(job, AnalysisOptions(essentia=essentia, lufs=essentia))
    assert result.key_camelot == "8A"
    assert result.key_source == ("essentia:edma" if essentia else "librosa:kk")
    assert (result.intro_bars, result.outro_bars) == (8, 16)
    assert result.structure_source == "rekordbox_grid"
    assert result.bpm_detected == pytest.approx(124, rel=0.03)
    assert (result.loudness_lufs is not None) == essentia
    assert (result.danceability is not None) == essentia


def test_structure_without_grid_uses_tag_tempo(audio_dir: Path) -> None:
    # The beat tracker halves this sparse track's tempo; the tag BPM keeps bars right.
    job = AnalysisJob("2", str(audio_dir / "sparse.wav"), (), 124.0)
    result = analyze_file(job, AnalysisOptions(essentia=False))
    assert result.structure_source == "librosa_beats"
    assert (result.intro_bars, result.outro_bars) == (16, 8)
    assert result.bars == 48


def test_energy_features_separate_busy_from_sparse(audio_dir: Path) -> None:
    opts = AnalysisOptions(essentia=False)
    busy = analyze_file(AnalysisJob("1", str(audio_dir / "busy.wav"), GRID, 124.0), opts)
    sparse = analyze_file(AnalysisJob("2", str(audio_dir / "sparse.wav"), GRID, 124.0), opts)
    for feature in ("loudness_db", "onset_rate", "spectral_centroid_hz", "spectral_flux"):
        assert getattr(busy, feature) > getattr(sparse, feature), feature


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="cannot read"):
        analyze_file(AnalysisJob("x", str(tmp_path / "missing.wav")))
    short = tmp_path / "short.wav"
    render_track(short, intro_bars=0, main_bars=4, outro_bars=0)
    with pytest.raises(AnalysisError, match="shorter than"):
        analyze_file(AnalysisJob("x", str(short)), AnalysisOptions(essentia=False))
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all" * 100)
    with pytest.raises(AnalysisError, match="could not decode"):
        analyze_file(AnalysisJob("x", str(junk)), AnalysisOptions(essentia=False))


def test_analyze_many_in_parallel(audio_dir: Path, tmp_path: Path) -> None:
    jobs = [
        AnalysisJob("1", str(audio_dir / "busy.wav"), GRID, 124.0),
        AnalysisJob("2", str(audio_dir / "sparse.wav"), GRID, 124.0),
        AnalysisJob("3", str(tmp_path / "missing.wav")),
    ]
    results = dict(analyze_many(jobs, AnalysisOptions(essentia=False), workers=2))
    by_id = {job.track_id: r for job, r in results.items()}
    assert by_id["1"].key_camelot == "8A"  # type: ignore[union-attr]
    assert by_id["2"].intro_bars == 16  # type: ignore[union-attr]
    assert isinstance(by_id["3"], AnalysisError)


def test_cli_analyze_then_use(audio_dir: Path, tmp_path: Path) -> None:
    def loc(name: str) -> str:
        return "file://localhost" + quote(str(audio_dir / name))

    xml = tmp_path / "collection.xml"
    write_collection(
        xml,
        [
            SynthTrack("1", "Kaya Sol", "Busy", 124.0, "Am", "Afro House", None, 3,
                       location=loc("busy.wav"), total_time=95),
            SynthTrack("2", "Dex Marlo", "Sparse", 124.0, "C#m", "Afro House", None, 3,
                       location=loc("sparse.wav"), total_time=95),
            SynthTrack("3", "Nils Lind", "Gone", 124.0, "Am", "Afro House", None, 3,
                       location=loc("gone.wav"), total_time=95),
        ],
    )  # fmt: skip
    runner = CliRunner()
    args = ["analyze", str(xml), "--workers", "1", "--no-essentia", "--json"]
    first = json.loads(runner.invoke(app, args).output)
    assert first["analyzed"] == 2
    assert [m["id"] for m in first["missing_files"]] == ["3"]
    assert first["coverage"]["key_agree"] == 1 and first["coverage"]["key_disagree"] == 1
    assert first["key_disagreements"][0] == {
        "id": "2", "track": "Dex Marlo - Sparse", "tag": "12A", "detected": "8A",
    }  # fmt: skip
    assert json.loads(runner.invoke(app, args).output)["analyzed"] == 0  # cached

    suggest = json.loads(runner.invoke(app, ["suggest", str(xml), "--id", "1", "--json"]).output)
    seed = suggest["seed"]
    assert seed["energy_source"] == "audio" and seed["outro_bars"] == 16
    (sparse,) = [s["track"] for s in suggest["suggestions"] if s["track"]["id"] == "2"]
    assert sparse["detected_key"] == "8A" and sparse["key_confidence"] == 0.4

    plain = json.loads(
        runner.invoke(app, ["suggest", str(xml), "--id", "1", "--no-analysis", "--json"]).output
    )
    assert plain["seed"]["energy_source"] == "rating"

    info = json.loads(runner.invoke(app, ["info", str(xml), "--json"]).output)
    assert info["analysis"]["analyzed"] == 2 and info["coverage"]["intro_outro"] == 2

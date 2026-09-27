"""Phase 5 on a synthesized DJ mix with known transitions. Skipped without the audio extra."""

import json
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("librosa")

from audio_synth import crossfade_mix, render_track
from synth import SynthTrack, write_collection
from typer.testing import CliRunner

from setsmith.analysis.audio import AnalysisError
from setsmith.analysis.liveset import LiveSet, analyze_liveset, mix_features
from setsmith.analysis.tracklist import match_tracklist, parse_tracklist
from setsmith.cli import app
from setsmith.io.rekordbox_xml import load_collection

pytestmark = pytest.mark.audio

BPM = 124.0
BAR = 4 * 60 / BPM
# Known truth: A plays bars 0-48 of the mix; B enters at mix bar 32 from its own bar 0
# (16-bar overlap) and plays 40 bars; C enters at mix bar 64 from its own bar 8 (8-bar
# overlap, cue-in 15.5s) and plays to the end.
TRACKLIST = "01. Kaya Sol - Alpha\n02. Moana Reyes - Beta\n03. Nils Lind - Gamma (Original Mix)\n"
TIMED = "0:00 Kaya Sol - Alpha\n1:02 Moana Reyes - Beta\n2:04 Nils Lind - Gamma\n"


@pytest.fixture(scope="module")
def mixset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("mixset")
    for name, transpose, seed, kick, sixteenths in [
        ("A", 0, 1, 50.0, False), ("B", 5, 2, 62.0, True), ("C", -3, 3, 42.0, False),
    ]:  # fmt: skip
        render_track(
            d / f"{name}.wav", intro_bars=8, main_bars=32, outro_bars=8, transpose=transpose,
            seed=seed, kick_hz=kick, sixteenth_hats=sixteenths,
        )  # fmt: skip
    crossfade_mix(
        d / "mix.wav",
        [
            (d / "A.wav", 0, 0, 48 * BAR),
            (d / "B.wav", 32 * BAR, 0, 40 * BAR),
            (d / "C.wav", 64 * BAR, 8 * BAR, 40 * BAR),
        ],
    )
    xml = d / "collection.xml"
    write_collection(
        xml,
        [
            SynthTrack(tid, artist, title, BPM, key, "Afro House", 5, 3, total_time=95,
                       location="file://localhost" + quote(str(d / f"{f}.wav")))
            for tid, artist, title, key, f in [
                ("1", "Kaya Sol", "Alpha", "Am", "A"),
                ("2", "Moana Reyes", "Beta", "Dm", "B"),
                ("3", "Nils Lind", "Gamma", "F#m", "C"),
            ]
        ],
    )  # fmt: skip
    # The synthetic tracks start on a downbeat at 0.0s.
    xml.write_text(xml.read_text().replace('Inizio="0.050"', 'Inizio="0.000"'))
    (d / "tracklist.txt").write_text(TRACKLIST)
    return {"dir": d, "mix": d / "mix.wav", "xml": xml, "tracklist": d / "tracklist.txt"}


def run(mixset: dict[str, Path], text: str, *, align: bool = True) -> LiveSet:
    col = load_collection(mixset["xml"])
    matches = match_tracklist(col, parse_tracklist(text))
    return analyze_liveset(
        "test", "tracklist.txt", matches, audio=mixset["mix"], align=align, use_essentia=False
    )


def test_windows_follow_tempo_and_keys(mixset: dict[str, Path]) -> None:
    mix = mix_features(mixset["mix"], use_essentia=False)
    assert mix.duration_s == pytest.approx(104 * BAR, abs=1.0)
    assert all(abs(w.bpm - BPM) < 1.0 for w in mix.windows if w.bpm)
    assert mix.windows[0].key == "8A"  # A minor
    assert mix.windows[-1].key == "11A"  # C is transposed to F# minor


def test_dtw_measures_transitions(mixset: dict[str, Path]) -> None:
    ls = run(mixset, TRACKLIST)
    assert [s.method for s in ls.spans] == ["dtw", "dtw", "dtw"]
    ab, bc = ls.transitions
    assert (ab.key_move, bc.key_move) == ("adjacent", "clash")  # 8A->7A, 7A->11A
    assert ab.overlap_bars == pytest.approx(16, abs=2)
    assert bc.overlap_bars == pytest.approx(8, abs=2)
    assert bc.cue_in_s == pytest.approx(8 * BAR, abs=2 * BAR)  # C cued in at its bar 8
    assert bc.cue_out_s == pytest.approx(40 * BAR, abs=2 * BAR)  # B cued out at its bar 40
    assert ab.cue_in_s == pytest.approx(0, abs=BAR)


def test_timestamps_without_alignment(mixset: dict[str, Path]) -> None:
    ls = run(mixset, TIMED, align=False)
    assert [s.method for s in ls.spans] == ["timestamps"] * 3
    assert [s.mix_start_s for s in ls.spans] == [0.0, 62.0, 124.0]
    assert all(t.overlap_bars is None for t in ls.transitions)  # nothing measured


def test_novelty_without_timestamps_or_alignment(mixset: dict[str, Path]) -> None:
    ls = run(mixset, TRACKLIST, align=False)
    assert [s.method for s in ls.spans] == ["novelty"] * 3
    starts = [s.mix_start_s for s in ls.spans]
    assert starts == sorted(starts) and starts[0] == 0.0


def test_unreadable_recording(tmp_path: Path) -> None:
    junk = tmp_path / "set.mp3"
    junk.write_bytes(b"not audio" * 100)
    with pytest.raises(AnalysisError, match="cannot stream"):
        mix_features(junk)


def test_cli_with_recording(
    mixset: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SETSMITH_STYLES_DIR", str(tmp_path / "styles"))
    result = CliRunner().invoke(
        app,
        [
            "liveset", "analyze", str(mixset["xml"]), str(mixset["tracklist"]),
            "--audio", str(mixset["mix"]), "--no-essentia", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    bars = data["stats"]["transition_bars"]
    assert bars == {"16": 1, "8": 1}
    draft = data["draft_profile"]
    assert draft["transition_length_bars"]["16"] == 0.5
    assert "recording: mix.wav" in draft["sources"]
    assert data["liveset"]["duration_s"] == pytest.approx(104 * BAR, abs=1.0)

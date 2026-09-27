"""Phase 5 logic without audio: transitions, stats, draft profiles, learning, CLI."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from setsmith.analysis.learn import (
    LearnedPreferences,
    blended_config,
    history_pairs,
    learn,
    tempo_band_label,
)
from setsmith.analysis.liveset import (
    LiveSet,
    MatchRecord,
    Span,
    Window,
    build_transitions,
    compute_stats,
    draft_profile,
    played_region,
    snap_bars,
)
from setsmith.cli import app
from setsmith.keys.camelot import KeyMove
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.store import Store
from setsmith.styles.profile import StyleProfile

runner = CliRunner()


def rec(position: int, key: str | None = "8A", bpm: float | None = 124.0, **kw: Any) -> MatchRecord:
    base: dict[str, Any] = {
        "raw": f"line {position}", "artist": f"Artist {position}", "title": f"Title {position}",
        "start_s": None, "unknown": False, "together": False, "track_id": str(position),
        "track_path": f"/m/{position}.mp3",
        "track_display": f"Artist {position} - Title {position}",
        "score": 1.0, "bpm": bpm, "camelot": key, "genre": "Afro House", "energy": 5.0,
    }  # fmt: skip
    base.update(kw)
    return MatchRecord(position=position, **base)


def test_build_transitions_moves_and_gaps() -> None:
    records = [
        rec(1, "8A", 122.0),
        rec(2, "9A", 124.0),
        rec(3, "9A", 124.0, together=True),  # layered: not a transition
        rec(4, "10B", 128.0),
        rec(5, track_id=None),  # unmatched: breaks the chain
        rec(6, "10B", 128.0),
    ]
    trs = build_transitions(records)
    assert [(t.from_position, t.to_position, t.key_move) for t in trs] == [
        (1, 2, "adjacent"),
        (2, 4, "diagonal"),
    ]
    assert trs[0].tempo_change_pct == pytest.approx(100 * 2 / 124, abs=0.01)
    assert trs[0].overlap_bars is None and trs[0].method == "tags"


def test_build_transitions_with_aligned_spans() -> None:
    records = [rec(1), rec(2)]
    spans = [
        Span(1, 0.0, 120.0, 0.0, 120.0, method="dtw"),
        Span(2, 89.0, 300.0, 16.0, 227.0, method="dtw"),
    ]
    (t,) = build_transitions(records, spans, bpm_at=lambda _: 124.0)
    assert t.overlap_s == 31.0
    assert t.overlap_bars == pytest.approx(31 * 124 / 60 / 4, abs=0.1)  # ~16 bars
    assert (t.cue_out_s, t.cue_in_s) == (120.0, 16.0)
    assert t.estimated_type == "long_blend"
    approx = [Span(1, 0.0, 120.0), Span(2, 110.0, 300.0)]  # timestamps: no overlap measured
    assert build_transitions(records, approx)[0].overlap_bars is None


@pytest.mark.parametrize(
    ("bars", "snapped"), [(2.0, None), (5.0, 8), (12.0, 16), (16.7, 16), (40.0, 32), (50.0, 64)]
)
def test_snap_bars(bars: float, snapped: int | None) -> None:
    assert snap_bars(bars) == snapped


def test_played_region() -> None:
    wander = [(0, 0), (1, 0), (2, 0)]
    diagonal = [(2 + k, 1 + k) for k in range(60)]
    assert played_region(wander + diagonal) == (3, 62)  # the diagonal pairs
    assert played_region([(k, k) for k in range(10)]) is None  # too short


def liveset(**kw: Any) -> LiveSet:
    records = [
        rec(1, "8A", 122.0, energy=4.0, label="L1"),
        rec(2, "9A", 123.0, energy=6.0, vocal=True, label="L1"),
        rec(3, "10A", 124.0, energy=8.0, genre="Melodic House & Techno"),
        rec(4, "10B", 125.0, energy=7.0, vocal=False),
    ]
    ls = LiveSet("Test set", "/sets/test.txt", None, None, records, **kw)
    ls.transitions = build_transitions(records, ls.spans, None)
    return ls


def test_stats_and_draft_from_tracklist_only() -> None:
    ls = liveset()
    stats = compute_stats(ls)
    assert stats.matched == 4
    assert stats.key_moves == {"adjacent": 2, "relative": 1}
    assert stats.minor_share == 0.75 and stats.vocal_share == 0.5
    assert stats.energy == [(0.0, 4.0), (1 / 3, 6.0), (2 / 3, 8.0), (1.0, 7.0)]
    draft = draft_profile(ls)
    profile = StyleProfile.model_validate(draft)  # a valid, loadable profile
    assert profile.name == "Draft from Test set"
    assert profile.allowed_key_moves[KeyMove.ADJACENT] == 1.0
    assert profile.allowed_key_moves[KeyMove.RELATIVE] == pytest.approx(2 / 3, abs=0.01)
    assert profile.allowed_key_moves[KeyMove.DIAGONAL] == pytest.approx(1 / 3, abs=0.01)
    assert profile.genre_weights == {"afro house": 1.0, "melodic house & techno": 0.33}
    assert "Setsmith defaults" in profile.notes  # nothing measured
    assert profile.reference_labels == ["L1"]
    lo, hi = profile.bpm_band
    assert lo < hi and 121 <= lo <= 123 and 124 <= hi <= 126


def test_draft_uses_measured_transitions_and_windows() -> None:
    windows = [Window(i * 30.0, 124.0, "8A", -20.0 + i) for i in range(8)]
    ls = liveset(windows=windows)
    ls.spans = [
        Span(1, 0, 100, 0, 100, "dtw"), Span(2, 70, 220, 0, 150, "dtw"),
        Span(3, 205, 400, 0, 195, "dtw"), Span(4, 395, 500, 0, 105, "dtw"),
    ]  # fmt: skip
    ls.transitions = build_transitions(ls.matches, ls.spans, lambda _: 124.0)
    draft = draft_profile(ls)
    StyleProfile.model_validate(draft)
    lengths = draft["transition_length_bars"]
    assert sum(lengths.values()) == pytest.approx(1.0)
    assert lengths["16"] > 0 and lengths["8"] > 0  # ~15.5 and ~7.8 bar overlaps
    assert draft["energy_curve"][0][1] < draft["energy_curve"][-1][1]  # rising loudness


def test_liveset_round_trip() -> None:
    ls = liveset(windows=[Window(0.0, 124.0, "8A", -12.0)], warnings=["x"])
    again = LiveSet.from_dict(json.loads(json.dumps(ls.to_dict())))
    assert again == ls


# ---------------------------------------------------------------- learning


def test_tempo_band_label() -> None:
    bands = DEFAULT_CONFIG.tempo.bands
    assert tempo_band_label(1.0, bands) == "<=2%"
    assert tempo_band_label(5.0, bands) == "<=6%"
    assert tempo_band_label(20.0, bands) == "beyond"


def test_learn_and_blend() -> None:
    a = Track(id="a", camelot="8A", bpm=124.0)
    b = Track(id="b", camelot="9A", bpm=124.0)
    c = Track(id="c", camelot="3A", bpm=130.0)
    pairs = history_pairs(
        [["/a", "/b", "/missing", "/c"], ["/b", "/c"]], {"/a": a, "/b": b, "/c": c}
    )
    assert [(x.id, y.id) for x, y in pairs] == [("a", "b"), ("b", "c")]
    prefs = learn(pairs, [("semitone_lift", 4.0)], prior=6)
    assert prefs.key_moves == {"adjacent": 1, "clash": 1, "semitone_lift": 1}
    assert prefs.sources == {"history": 2, "livesets": 1}
    cfg = blended_config(prefs)
    weight = 3 / (3 + 6)
    # adjacent: default 0.9, learned 1/1 = 1.0
    adjacent = cfg.harmonic.move_scores[KeyMove.ADJACENT]
    assert adjacent == pytest.approx(0.9 * (1 - weight) + weight, abs=1e-3)
    # same was never played: pulled toward 0
    assert cfg.harmonic.move_scores[KeyMove.SAME] == pytest.approx(1.0 * (1 - weight), abs=1e-3)
    # the original config is untouched
    assert DEFAULT_CONFIG.harmonic.move_scores[KeyMove.SAME] == 1.0
    assert LearnedPreferences.from_dict(prefs.to_dict()) == prefs


def test_blend_without_data_is_default() -> None:
    cfg = blended_config(LearnedPreferences())
    assert cfg.harmonic.move_scores == DEFAULT_CONFIG.harmonic.move_scores
    assert cfg.tempo.bands == DEFAULT_CONFIG.tempo.bands


# ---------------------------------------------------------------- store and CLI


def test_store_livesets(tmp_path: Path) -> None:
    with Store(tmp_path / "s.db") as store:
        first = store.save_liveset(liveset().to_dict())
        second = store.save_liveset(liveset().to_dict())
        assert [i for i, _, _ in store.list_livesets()] == [first, second]
        data = store.get_liveset(first)
        assert data is not None and LiveSet.from_dict(data).name == "Test set"
        assert store.delete_liveset(first) and not store.delete_liveset(first)
        assert store.get_liveset(first) is None


def test_cli_liveset_and_learn(
    fixture_5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SETSMITH_STYLES_DIR", str(tmp_path / "styles"))
    tracklist = tmp_path / "friday.txt"
    tracklist.write_text(
        "01. Kaya Sol - Sunwater\n02. Moana Reyes - Lanterns\n03. Nils Odegaard - Glass Harbour\n"
        "04. Someone - Unknown Song\n",
        encoding="utf-8",
    )
    nothing = runner.invoke(app, ["suggest", str(fixture_5), "--id", "101", "--learned"])
    assert nothing.exit_code == 2

    result = runner.invoke(
        app,
        [
            "liveset",
            "analyze",
            str(fixture_5),
            str(tracklist),
            "--save-profile",
            "friday",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["id"] == 1 and data["stats"]["matched"] == 3 and data["stats"]["unmatched"] == 1
    assert data["stats"]["key_moves"] == {"adjacent": 1, "diagonal": 1}
    assert Path(data["saved_profile"]).name == "friday.json"
    assert runner.invoke(app, ["styles", "show", "friday"]).exit_code == 0

    listed = json.loads(runner.invoke(app, ["liveset", "list", "--json"]).output)
    assert [x["name"] for x in listed] == ["friday"]
    shown = runner.invoke(app, ["liveset", "show", "1"])
    assert shown.exit_code == 0 and "Lanterns" in shown.output
    again = runner.invoke(app, ["liveset", "profile", "1", "--save-profile", "friday"])
    assert again.exit_code == 3  # refuses to replace without --force
    assert runner.invoke(app, ["liveset", "show", "9"]).exit_code == 2

    learned = json.loads(runner.invoke(app, ["learn", str(fixture_5), "--json"]).output)
    assert learned["sources"] == {"history": 0, "livesets": 2}
    suggest = runner.invoke(app, ["suggest", str(fixture_5), "--id", "101", "--learned", "--json"])
    assert suggest.exit_code == 0, suggest.output

    assert runner.invoke(app, ["liveset", "delete", "1"]).exit_code == 0
    assert runner.invoke(app, ["liveset", "delete", "1"]).exit_code == 2

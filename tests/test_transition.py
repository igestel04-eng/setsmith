from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from setsmith.keys.camelot import all_keys, normalize_key
from setsmith.model.track import Track
from setsmith.scoring.transition import (
    Flag,
    TransitionType,
    base_upper_bound,
    harmonic_score,
    score_transition,
    tempo_score,
)
from setsmith.scoring.weights import DEFAULT_CONFIG, ComponentWeights


def mk(tid: str = "x", key: str | None = "8A", **kw: Any) -> Track:
    defaults: dict[str, Any] = {"bpm": 124.0, "energy": 6.0, "genre": "Afro House"}
    defaults.update(kw)
    return Track(id=tid, key_raw=key, camelot=normalize_key(key), **defaults)


def comp(a: Track, b: Track, name: str, **kw: Any) -> float:
    return score_transition(a, b, **kw).components[name].score


# ---------------------------------------------------------------- tempo


@pytest.mark.parametrize(
    ("bpm_b", "expected", "flag"),
    [
        (124.0, 1.0, None),
        (126.0, 1.0, None),  # 1.6%
        (128.0, 0.8, None),  # 3.1%
        (130.0, 0.5, Flag.TEMPO_STRETCH),  # 4.6%
        (134.0, 0.2, Flag.TEMPO_STRETCH),  # 7.5%
        (136.0, 0.0, Flag.TEMPO_CLASH),  # 8.8%
    ],
)
def test_tempo_bands(bpm_b: float, expected: float, flag: Flag | None) -> None:
    s = score_transition(mk("a", bpm=124.0), mk("b", bpm=bpm_b))
    assert s.components["tempo"].score == expected
    if flag:
        assert flag in s.flags
    else:
        assert Flag.TEMPO_STRETCH not in s.flags and Flag.TEMPO_CLASH not in s.flags


@pytest.mark.parametrize(("a", "b"), [(85.0, 170.0), (170.0, 85.0), (64.0, 128.0), (128.0, 64.0)])
def test_half_double_time(a: float, b: float) -> None:
    s = score_transition(mk("a", bpm=a), mk("b", bpm=b))
    assert s.components["tempo"].score == 1.0
    assert Flag.HALF_DOUBLE_TIME in s.flags
    assert "time" in s.components["tempo"].reason


def test_direct_match_not_flagged_half_double() -> None:
    s = score_transition(mk("a", bpm=128.0), mk("b", bpm=128.0))
    assert Flag.HALF_DOUBLE_TIME not in s.flags


def test_missing_bpm() -> None:
    s = score_transition(mk("a", bpm=None), mk("b"))
    assert s.components["tempo"].score == 0.5
    assert Flag.MISSING_BPM in s.flags


def test_tempo_clash_forces_cut_or_echo() -> None:
    s = score_transition(mk("a", bpm=110.0), mk("b", bpm=128.0))
    assert s.suggested_type in (TransitionType.CUT, TransitionType.ECHO_OUT)


# ---------------------------------------------------------------- harmonic


@pytest.mark.parametrize(
    ("ka", "kb", "expected", "flag"),
    [
        ("8A", "8A", 1.0, None),
        ("8A", "9A", 0.9, None),
        ("8A", "7A", 0.9, None),
        ("8A", "8B", 0.85, None),
        ("8A", "10A", 0.65, Flag.ENERGY_BOOST_KEY),
        ("8A", "9B", 0.5, None),
        ("8A", "3A", 0.5, Flag.SEMITONE_LIFT),
        ("8A", "2A", 0.1, Flag.KEY_CLASH),
        ("Am", "Em", 0.9, None),  # classic notation end to end
    ],
)
def test_harmonic_moves(ka: str, kb: str, expected: float, flag: Flag | None) -> None:
    s = score_transition(mk("a", key=ka), mk("b", key=kb))
    assert s.components["harmonic"].score == pytest.approx(expected)
    if flag:
        assert flag in s.flags


def test_missing_key() -> None:
    s = score_transition(mk("a", key=None), mk("b"))
    assert s.components["harmonic"].score == 0.5
    assert Flag.MISSING_KEY in s.flags


def test_key_confidence_pulls_toward_half() -> None:
    a = mk("a", key="8A")
    b = mk("b", key="8A")
    b.key_confidence = 0.6
    assert comp(a, b, "harmonic") == pytest.approx(0.5 + 0.5 * 0.6)
    assert Flag.LOW_KEY_CONFIDENCE in score_transition(a, b).flags
    clash = mk("c", key="2A")
    clash.key_confidence = 0.6
    assert comp(a, clash, "harmonic") == pytest.approx(0.5 + (0.1 - 0.5) * 0.6)


def test_semitone_lift_and_clash_suggest_cut() -> None:
    for kb in ("3A", "2A"):
        s = score_transition(mk("a", key="8A"), mk("b", key=kb))
        assert s.suggested_type == TransitionType.CUT
    sparse = mk("b", key="2A", intro_bars=32)
    assert score_transition(mk("a"), sparse).suggested_type == TransitionType.ECHO_OUT


# ---------------------------------------------------------------- energy


@pytest.mark.parametrize(
    ("eb", "target", "expected"),
    [(6, 0, 1.0), (7, 0, 1.0), (8, 0, 0.7), (3, 0, 0.4), (1, 0, 0.1), (8, 2, 1.0), (6, 2, 0.7)],
)
def test_energy_bands(eb: float, target: float, expected: float) -> None:
    assert comp(mk("a", energy=6.0), mk("b", energy=eb), "energy", energy_target=target) == expected


def test_missing_energy() -> None:
    s = score_transition(mk("a", energy=None), mk("b"))
    assert s.components["energy"].score == 0.5
    assert Flag.MISSING_ENERGY in s.flags


def test_energy_jump_suggests_drop_swap() -> None:
    s = score_transition(mk("a", energy=4.0), mk("b", energy=8.0))
    assert s.suggested_type == TransitionType.DROP_SWAP


# ---------------------------------------------------------------- genre


@pytest.mark.parametrize(
    ("ga", "gb", "expected"),
    [
        ("Afro House", "afro-house", 1.0),
        ("Afro House", "Organic House", 0.7),
        ("Melodic House", "Deep House", 0.7),  # alias -> melodic house & techno
        ("Tech House", "Minimal / Deep Tech", 0.7),
        ("House", "Tech House", 0.7),
        ("Afro House", "Techno", 0.3),
        ("Deep House, Afro House", "Afro House", 1.0),
        ("", "House", 0.5),
    ],
)
def test_genre(ga: str, gb: str, expected: float) -> None:
    assert comp(mk("a", genre=ga), mk("b", genre=gb), "genre_style") == expected


# ---------------------------------------------------------------- extras


def test_extras_no_data_is_neutral() -> None:
    assert comp(mk("a"), mk("b"), "extras") == 0.5


def test_extras_rating() -> None:
    assert comp(mk("a"), mk("b", rating=4), "extras") == pytest.approx(0.8)


def test_vocal_clash_in_long_blend() -> None:
    a = mk("a", vocal=True)
    b = mk("b", vocal=True, rating=5)
    s = score_transition(a, b)
    assert s.suggested_type == TransitionType.LONG_BLEND
    assert Flag.VOCAL_CLASH in s.flags
    assert s.components["extras"].score == pytest.approx((0.3 + 1.0) / 2)


def test_arrangement_fit() -> None:
    long_ = score_transition(mk("a", outro_bars=32), mk("b", intro_bars=16))
    assert long_.components["extras"].score == 1.0
    assert long_.suggested_type == TransitionType.LONG_BLEND
    short = score_transition(mk("a", outro_bars=4), mk("b", intro_bars=32))
    assert short.components["extras"].score == 0.4
    assert Flag.SHORT_ARRANGEMENT in short.flags
    assert short.suggested_type in (TransitionType.CUT, TransitionType.DROP_SWAP)


# ---------------------------------------------------------------- type and total


def test_perfect_match_is_long_blend_32() -> None:
    s = score_transition(mk("a"), mk("b"))
    assert s.suggested_type == TransitionType.LONG_BLEND
    assert s.suggested_length_bars == 32
    # everything perfect except extras (no data -> 0.5)
    assert s.total == pytest.approx(100 * (1 - 0.10) + 100 * 0.10 * 0.5)


def test_middling_pair_gets_filter_sweep() -> None:
    s = score_transition(mk("a", bpm=124.0), mk("b", bpm=129.0, key="9B"))
    assert s.suggested_type == TransitionType.FILTER_SWEEP
    assert s.suggested_length_bars == 16


def test_every_component_has_a_reason() -> None:
    s = score_transition(mk("a"), mk("b", key="10A", bpm=130.0))
    assert set(s.components) == {"harmonic", "tempo", "energy", "genre_style", "extras"}
    assert all(c.reason for c in s.components.values())
    assert s.type_reason
    lines = s.explain()
    assert len(lines) == 6
    d = s.to_dict()
    assert d["flags"] == ["energy_boost_key", "tempo_stretch"]
    assert d["components"]["harmonic"]["reason"].startswith("8A -> 10A")


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ComponentWeights(harmonic=0.5)


def test_scores_are_directional() -> None:
    up = score_transition(mk("a", key="8A"), mk("b", key="10A"))
    down = score_transition(mk("b", key="10A"), mk("a", key="8A"))
    assert up.total != down.total


# ---------------------------------------------------------------- properties

KEY_TEXT = [str(k) for k in all_keys()]
_opt_bpm = st.one_of(st.none(), st.floats(min_value=40, max_value=220))
_opt_energy = st.one_of(st.none(), st.floats(min_value=1, max_value=10))
_opt_bars = st.one_of(st.none(), st.integers(min_value=0, max_value=128))


@st.composite
def tracks(draw: st.DrawFn, *, full: bool = False) -> Track:
    key = draw(
        st.sampled_from(KEY_TEXT) if full else st.one_of(st.none(), st.sampled_from(KEY_TEXT))
    )
    return Track(
        id=draw(st.text(min_size=1, max_size=4)),
        bpm=draw(st.floats(min_value=40, max_value=220) if full else _opt_bpm),
        key_raw=key,
        camelot=normalize_key(key),
        key_confidence=1.0 if full else draw(st.floats(min_value=0, max_value=1)),
        energy=draw(_opt_energy),
        genre=draw(
            st.sampled_from(["", "Afro House", "Tech House", "techno", "Deep House, House"])
        ),
        rating=draw(st.integers(min_value=0, max_value=5)),
        vocal=draw(st.one_of(st.none(), st.booleans())),
        intro_bars=draw(_opt_bars),
        outro_bars=draw(_opt_bars),
    )


@given(tracks(), tracks(), st.one_of(st.none(), st.floats(min_value=-9, max_value=9)))
@settings(max_examples=500)
def test_property_scores_in_range(a: Track, b: Track, target: float | None) -> None:
    s = score_transition(a, b, energy_target=target)
    assert 0.0 <= s.total <= 100.0
    for c in s.components.values():
        assert 0.0 <= c.score <= 1.0
    assert s.suggested_length_bars in DEFAULT_CONFIG.transition.allowed_lengths_bars
    if Flag.TEMPO_CLASH in s.flags:
        assert s.suggested_type in (TransitionType.CUT, TransitionType.ECHO_OUT)


@given(tracks(full=True))
def test_property_self_pair_harmonic_and_tempo(t: Track) -> None:
    s = score_transition(t, t)
    assert 100 * s.components["harmonic"].score >= 95
    assert 100 * s.components["tempo"].score >= 95


@given(tracks(), tracks())
@settings(max_examples=500)
def test_property_upper_bound_never_undercounts(a: Track, b: Track) -> None:
    s = score_transition(a, b)
    base = s.total - s.components["energy"].points
    assert base_upper_bound(a, b) >= base - 1e-9
    assert harmonic_score(a, b) == s.components["harmonic"].score
    assert tempo_score(a, b) == s.components["tempo"].score

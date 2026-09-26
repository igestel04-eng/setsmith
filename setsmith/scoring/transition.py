"""Pairwise transition scoring: how well does track b follow track a?

Scores are directional (a -> b). Each component returns a 0-1 score and a plain-English
reason; the total is 100 x the weighted sum. All thresholds come from scoring/weights.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from setsmith.keys.camelot import KeyMove, camelot_step, classify_move, parse_key
from setsmith.model.track import Track
from setsmith.scoring.weights import (
    DEFAULT_CONFIG,
    ScoringConfig,
    band_score,
)

# Bump when scoring logic changes, so cached pair scores are recomputed.
SCORER_VERSION = 1


class TransitionType(StrEnum):
    LONG_BLEND = "long_blend"
    CUT = "cut"
    ECHO_OUT = "echo_out"
    FILTER_SWEEP = "filter_sweep"
    DROP_SWAP = "drop_swap"


class Flag(StrEnum):
    HALF_DOUBLE_TIME = "half_double_time"
    TEMPO_STRETCH = "tempo_stretch"
    TEMPO_CLASH = "tempo_clash"
    MISSING_BPM = "missing_bpm"
    VARIABLE_TEMPO = "variable_tempo"
    ENERGY_BOOST_KEY = "energy_boost_key"
    SEMITONE_LIFT = "semitone_lift"
    KEY_CLASH = "key_clash"
    MISSING_KEY = "missing_key"
    LOW_KEY_CONFIDENCE = "low_key_confidence"
    MISSING_ENERGY = "missing_energy"
    MISSING_GENRE = "missing_genre"
    VOCAL_CLASH = "vocal_clash"
    SHORT_ARRANGEMENT = "short_arrangement"


@dataclass(frozen=True, slots=True)
class ComponentScore:
    score: float  # 0-1
    weight: float
    reason: str

    @property
    def points(self) -> float:
        """Contribution to the 0-100 total."""
        return 100 * self.weight * self.score


@dataclass(frozen=True, slots=True)
class TransitionScore:
    a_id: str
    b_id: str
    total: float  # 0-100
    components: dict[str, ComponentScore]
    flags: list[Flag]
    suggested_type: TransitionType
    suggested_length_bars: int
    type_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_id": self.a_id,
            "b_id": self.b_id,
            "total": round(self.total, 1),
            "components": {
                name: {
                    "score": round(c.score, 3),
                    "weight": c.weight,
                    "points": round(c.points, 1),
                    "reason": c.reason,
                }
                for name, c in self.components.items()
            },
            "flags": [f.value for f in self.flags],
            "suggested_type": self.suggested_type.value,
            "suggested_length_bars": self.suggested_length_bars,
            "type_reason": self.type_reason,
        }

    def explain(self) -> list[str]:
        lines = [
            f"{name:<12} {c.score:4.2f} x {c.weight:.2f} = {c.points:4.1f}  {c.reason}"
            for name, c in self.components.items()
        ]
        lines.append(
            f"{'transition':<12} {self.suggested_type.value}, {self.suggested_length_bars} bars: "
            f"{self.type_reason}"
        )
        return lines


# ---------------------------------------------------------------- components


@dataclass(slots=True)
class _Part:
    score: float
    reason: str
    flags: list[Flag] = field(default_factory=list)


@dataclass(slots=True)
class _TempoPart(_Part):
    pct: float | None = None  # pitch change needed on b, in percent


def tempo_match(a_bpm: float, b_bpm: float) -> tuple[float, float, str]:
    """(tempo b plays at, percent pitch change on b, "direct"/"double"/"half")."""
    # Direct first so an exact tie keeps the direct match.
    targets = ((a_bpm, "direct"), (a_bpm * 2, "double"), (a_bpm / 2, "half"))
    target, relation = min(targets, key=lambda tr: abs(tr[0] - b_bpm))
    return target, abs(target - b_bpm) / b_bpm * 100, relation


def tempo_score(a: Track, b: Track, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Tempo component alone (0-1)."""
    t = cfg.tempo
    if not a.bpm or not b.bpm:
        return t.missing_score
    return band_score(tempo_match(a.bpm, b.bpm)[1], t.bands, t.beyond_score)


def _tempo(a: Track, b: Track, cfg: ScoringConfig) -> _TempoPart:
    t = cfg.tempo
    if not a.bpm or not b.bpm:
        return _TempoPart(t.missing_score, "BPM unknown", [Flag.MISSING_BPM])

    a_bpm, b_bpm = a.bpm, b.bpm
    target, pct, relation = tempo_match(a_bpm, b_bpm)
    score = band_score(pct, t.bands, t.beyond_score)

    flags: list[Flag] = []
    if relation != "direct":
        flags.append(Flag.HALF_DOUBLE_TIME)
    if pct > t.clash_flag_above_pct:
        flags.append(Flag.TEMPO_CLASH)
    elif pct > t.stretch_flag_above_pct:
        flags.append(Flag.TEMPO_STRETCH)
    if a.variable_tempo or b.variable_tempo:
        flags.append(Flag.VARIABLE_TEMPO)

    detail = f"{a_bpm:g} -> {b_bpm:g} BPM"
    if relation != "direct":
        detail += f" ({relation}-time: b plays against {target:g})"
    reason = f"{detail}, needs {pct:.1f}% pitch change"
    if Flag.TEMPO_CLASH in flags:
        reason += ": too far to beatmatch, cut or echo out"
    return _TempoPart(score, reason, flags, pct)


@dataclass(slots=True)
class _HarmonicPart(_Part):
    move: KeyMove | None = None


_MOVE_LABEL = {
    KeyMove.SAME: "same key",
    KeyMove.ADJACENT: "adjacent",
    KeyMove.RELATIVE: "relative major/minor",
    KeyMove.ENERGY_BOOST: "energy boost",
    KeyMove.DIAGONAL: "diagonal",
    KeyMove.SEMITONE_LIFT: "semitone lift",
    KeyMove.CLASH: "key clash",
}
_MOVE_FLAG = {
    KeyMove.ENERGY_BOOST: Flag.ENERGY_BOOST_KEY,
    KeyMove.SEMITONE_LIFT: Flag.SEMITONE_LIFT,
    KeyMove.CLASH: Flag.KEY_CLASH,
}


def harmonic_score(a: Track, b: Track, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Harmonic component alone (0-1), including the key-confidence pull."""
    h = cfg.harmonic
    ka, kb = parse_key(a.camelot), parse_key(b.camelot)
    if ka is None or kb is None:
        return h.missing_score
    raw = h.move_scores[classify_move(ka, kb)]
    return h.neutral_score + (raw - h.neutral_score) * a.key_confidence * b.key_confidence


def base_upper_bound(a: Track, b: Track, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Cheap ceiling on the total minus the energy component: exact harmonic and tempo,
    with genre and extras assumed perfect. Lets searches skip full scoring of hopeless pairs.
    """
    w = cfg.weights
    return 100 * (
        w.harmonic * harmonic_score(a, b, cfg)
        + w.tempo * tempo_score(a, b, cfg)
        + w.genre_style
        + w.extras
    )


def _harmonic(a: Track, b: Track, cfg: ScoringConfig) -> _HarmonicPart:
    h = cfg.harmonic
    ka, kb = parse_key(a.camelot), parse_key(b.camelot)
    if ka is None or kb is None:
        return _HarmonicPart(h.missing_score, "key unknown", [Flag.MISSING_KEY])

    move = classify_move(ka, kb)
    confidence = a.key_confidence * b.key_confidence
    score = harmonic_score(a, b, cfg)

    flags = [_MOVE_FLAG[move]] if move in _MOVE_FLAG else []
    step = camelot_step(ka, kb)
    step_text = f", {step:+d}" if step else ""
    reason = f"{ka} -> {kb} ({_MOVE_LABEL[move]}{step_text})"
    if confidence < 1.0:
        reason += f", key confidence {confidence:.2f}"
        if confidence < h.low_confidence_flag_below:
            flags.append(Flag.LOW_KEY_CONFIDENCE)
    return _HarmonicPart(score, reason, flags, move)


@dataclass(slots=True)
class _EnergyPart(_Part):
    delta: float | None = None


def energy_score(a: Track, b: Track, target: float, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Energy component alone (0-1): how close b - a lands to the target change."""
    e = cfg.energy
    if a.energy is None or b.energy is None:
        return e.missing_score
    return band_score(abs(b.energy - a.energy - target), e.bands, e.beyond_score)


def _energy(a: Track, b: Track, target: float, cfg: ScoringConfig) -> _EnergyPart:
    if a.energy is None or b.energy is None:
        return _EnergyPart(cfg.energy.missing_score, "energy unknown", [Flag.MISSING_ENERGY])
    delta = b.energy - a.energy
    score = energy_score(a, b, target, cfg)
    reason = f"E{a.energy:g} -> E{b.energy:g} ({delta:+g}, target {target:+g})"
    return _EnergyPart(score, reason, [], delta)


def _genre(a: Track, b: Track, cfg: ScoringConfig) -> _Part:
    g = cfg.genre
    ga, gb = g.normalize(a.genre), g.normalize(b.genre)
    if not ga or not gb:
        return _Part(g.missing_score, "genre unknown", [Flag.MISSING_GENRE])
    if ga & gb:
        return _Part(g.same_score, f"same genre ({sorted(ga & gb)[0]})")
    for x in sorted(ga):
        for y in sorted(gb):
            if g.are_neighbors(x, y):
                return _Part(g.neighbor_score, f"neighboring genres ({x} -> {y})")
    return _Part(g.other_score, f"different genres ({a.genre} -> {b.genre})")


def _arrangement_known(a: Track, b: Track) -> bool:
    return a.outro_bars is not None and b.intro_bars is not None


def _extras(a: Track, b: Track, kind: TransitionType, cfg: ScoringConfig) -> _Part:
    x = cfg.extras
    subs: list[tuple[float, str]] = []
    flags: list[Flag] = []

    if a.vocal is not None and b.vocal is not None:
        if a.vocal and b.vocal and kind == TransitionType.LONG_BLEND:
            subs.append((x.vocal_clash_score, "both vocal in a long blend"))
            flags.append(Flag.VOCAL_CLASH)
        else:
            subs.append((x.vocal_ok_score, "no vocal overlap"))

    if b.rating > 0:
        subs.append((b.rating / x.rating_max, f"rated {b.rating}/{x.rating_max:g}"))

    if a.outro_bars is not None and b.intro_bars is not None:
        shortest = min(a.outro_bars, b.intro_bars)
        section = f"outro {a.outro_bars} / intro {b.intro_bars} bars"
        if shortest < x.arrangement_short_below_bars:
            subs.append((x.arrangement_short_score, f"{section}: short"))
            flags.append(Flag.SHORT_ARRANGEMENT)
        elif shortest >= x.arrangement_long_min_bars:
            subs.append((x.arrangement_good_score, section))
        else:
            subs.append((x.arrangement_mid_score, section))

    if not subs:
        return _Part(x.no_data_score, "no vocal, rating or arrangement data", flags)
    score = sum(s for s, _ in subs) / len(subs)
    return _Part(score, "; ".join(r for _, r in subs), flags)


# ---------------------------------------------------------------- transition type


def _snap_length(bars: int, cfg: ScoringConfig) -> int:
    allowed = cfg.transition.allowed_lengths_bars
    return min(allowed, key=lambda n: (abs(n - bars), n))


def _suggest_type(
    a: Track,
    b: Track,
    tempo: _TempoPart,
    harmonic: _HarmonicPart,
    energy: _EnergyPart,
    cfg: ScoringConfig,
) -> tuple[TransitionType, int, str]:
    tc = cfg.transition
    tempo_far = tempo.pct is not None and tempo.pct > cfg.tempo.blend_max_pct
    bad_move = harmonic.move if harmonic.move in (KeyMove.CLASH, KeyMove.SEMITONE_LIFT) else None

    if tempo_far or bad_move:
        reasons = (["tempo too far to blend"] if tempo_far else []) + (
            [_MOVE_LABEL[bad_move]] if bad_move else []
        )
        why = " and ".join(reasons)
        b_sparse = b.intro_bars is not None and b.intro_bars >= tc.sparse_intro_min_bars
        if b_sparse:
            return TransitionType.ECHO_OUT, tc.echo_out_bars, f"{why}; b starts sparse"
        return TransitionType.CUT, tc.cut_bars, f"{why}; cut on a phrase boundary"

    if energy.delta is not None and energy.delta >= tc.drop_swap_min_energy_jump:
        return TransitionType.DROP_SWAP, tc.drop_swap_bars, f"energy jumps {energy.delta:+g}"

    sections_ok = (a.outro_bars is None or a.outro_bars >= tc.long_blend_min_section_bars) and (
        b.intro_bars is None or b.intro_bars >= tc.long_blend_min_section_bars
    )
    tempo_close = tempo.pct is not None and tempo.pct <= cfg.tempo.long_blend_max_pct
    if tempo_close and harmonic.score >= cfg.harmonic.long_blend_min_score and sections_ok:
        return (
            TransitionType.LONG_BLEND,
            tc.long_blend_bars,
            "tempo and key compatible; swap basslines on a phrase boundary",
        )

    if _arrangement_known(a, b) and min(a.outro_bars or 0, b.intro_bars or 0) < (
        cfg.extras.arrangement_short_below_bars
    ):
        if energy.delta is None or energy.delta >= 0:
            return TransitionType.DROP_SWAP, tc.drop_swap_bars, "short intro/outro"
        return TransitionType.CUT, tc.cut_bars, "short intro/outro"

    missing = [
        label
        for label, part in (("BPM", tempo), ("key", harmonic))
        if Flag.MISSING_BPM in part.flags or Flag.MISSING_KEY in part.flags
    ]
    if missing:
        why = f"{' and '.join(missing)} unknown, keep the overlap short"
    else:
        why = "compatible but not close enough for a long blend"
    return TransitionType.FILTER_SWEEP, tc.filter_sweep_bars, why


# ---------------------------------------------------------------- public API


def score_transition(
    a: Track,
    b: Track,
    *,
    energy_target: float | None = None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> TransitionScore:
    """Score the transition from a (playing) to b (incoming).

    energy_target is the desired energy change (b - a) on the 1-10 scale; set generation
    passes the curve's target, suggestions default to holding energy steady.
    """
    target = cfg.energy.default_target_delta if energy_target is None else energy_target
    tempo = _tempo(a, b, cfg)
    harmonic = _harmonic(a, b, cfg)
    energy = _energy(a, b, target, cfg)
    genre = _genre(a, b, cfg)
    kind, bars, why = _suggest_type(a, b, tempo, harmonic, energy, cfg)
    extras = _extras(a, b, kind, cfg)

    w = cfg.weights
    components = {
        "harmonic": ComponentScore(harmonic.score, w.harmonic, harmonic.reason),
        "tempo": ComponentScore(tempo.score, w.tempo, tempo.reason),
        "energy": ComponentScore(energy.score, w.energy, energy.reason),
        "genre_style": ComponentScore(genre.score, w.genre_style, genre.reason),
        "extras": ComponentScore(extras.score, w.extras, extras.reason),
    }
    total = sum(c.points for c in components.values())
    flags = [*harmonic.flags, *tempo.flags, *energy.flags, *genre.flags, *extras.flags]
    return TransitionScore(
        a_id=a.id,
        b_id=b.id,
        total=max(0.0, min(100.0, total)),
        components=components,
        flags=flags,
        suggested_type=kind,
        suggested_length_bars=_snap_length(bars, cfg),
        type_reason=why,
    )

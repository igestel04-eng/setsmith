"""Default weights and thresholds. The single source of truth for every tunable number.

Bands are ``(max_value, score)`` pairs checked in order: the first band whose ``max_value``
is >= the measured distance wins, otherwise ``beyond_score`` applies.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from setsmith.keys.camelot import KeyMove

Bands = tuple[tuple[float, float], ...]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentWeights(_Config):
    harmonic: float = 0.30
    tempo: float = 0.25
    energy: float = 0.20
    genre_style: float = 0.15
    extras: float = 0.10

    @model_validator(mode="after")
    def _sum_to_one(self) -> ComponentWeights:
        total = self.harmonic + self.tempo + self.energy + self.genre_style + self.extras
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"component weights must sum to 1.0, got {total:.4f}")
        return self


class TempoConfig(_Config):
    # Percent pitch change the incoming track needs to match the outgoing tempo.
    bands: Bands = ((2.0, 1.0), (4.0, 0.8), (6.0, 0.5), (8.0, 0.2))
    beyond_score: float = 0.0
    missing_score: float = 0.5
    stretch_flag_above_pct: float = 4.0  # flag "tempo_stretch" above this
    clash_flag_above_pct: float = 8.0  # flag "tempo_clash" above this: cut/echo_out only
    blend_max_pct: float = 6.0  # above this, only cut or echo_out
    long_blend_max_pct: float = 4.0
    variable_tempo_min_bpm_diff: float = 0.01  # TEMPO markers closer than this count as equal


class HarmonicConfig(_Config):
    move_scores: dict[KeyMove, float] = {
        KeyMove.SAME: 1.0,
        KeyMove.ADJACENT: 0.9,
        KeyMove.RELATIVE: 0.85,
        KeyMove.ENERGY_BOOST: 0.65,
        KeyMove.DIAGONAL: 0.5,
        KeyMove.SEMITONE_LIFT: 0.5,
        KeyMove.CLASH: 0.1,
    }
    missing_score: float = 0.5
    # Low confidence pulls the score toward this value: 0.5 + (score - 0.5) x confidence.
    neutral_score: float = 0.5
    low_confidence_flag_below: float = 0.7
    long_blend_min_score: float = 0.85
    default_tag_confidence: float = 1.0  # confidence of a key read from Rekordbox tags


class EnergyConfig(_Config):
    # Distance between the actual energy delta and the target delta, on the 1-10 scale.
    bands: Bands = ((1.0, 1.0), (2.0, 0.7), (3.0, 0.4))
    beyond_score: float = 0.1
    missing_score: float = 0.5
    default_target_delta: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 10.0
    # Until Phase 3 audio analysis: energy = rating x this, when no "E7" comment tag exists.
    from_rating_multiplier: float = 2.0


class GenreConfig(_Config):
    same_score: float = 1.0
    neighbor_score: float = 0.7
    other_score: float = 0.3
    missing_score: float = 0.5
    style_profile_blend: float = 0.5  # Phase 4: share of the profile's genre weight
    # Genres in the same group are neighbors. A genre may belong to several groups.
    neighbor_groups: tuple[frozenset[str], ...] = (
        frozenset({"afro house", "melodic house & techno", "deep house", "organic house"}),
        frozenset({"house", "tech house", "minimal / deep tech"}),
        frozenset({"house", "deep house"}),
        frozenset({"indie dance", "melodic house & techno"}),
    )
    # Spelling variants mapped to one canonical name (after lowercasing and trimming).
    aliases: dict[str, str] = {
        "afro-house": "afro house",
        "afrohouse": "afro house",
        "melodic house": "melodic house & techno",
        "melodic techno": "melodic house & techno",
        "melodic house and techno": "melodic house & techno",
        "melodic house/techno": "melodic house & techno",
        "organic": "organic house",
        "organic house / downtempo": "organic house",
        "tech-house": "tech house",
        "techhouse": "tech house",
        "deep-house": "deep house",
        "minimal": "minimal / deep tech",
        "deep tech": "minimal / deep tech",
        "minimal/deep tech": "minimal / deep tech",
        "minimal deep tech": "minimal / deep tech",
        "indie-dance": "indie dance",
        "indie dance / nu disco": "indie dance",
    }
    # Characters that separate several genres in one Genre tag.
    multi_genre_separators: str = ",;"

    _cache: dict[str, frozenset[str]] = PrivateAttr(default_factory=dict)

    def normalize(self, raw: str) -> frozenset[str]:
        """Split a Genre tag into canonical lowercase genre names ("Afro-House" -> "afro house")."""
        cached = self._cache.get(raw)
        if cached is not None:
            return cached
        parts = [raw]
        for sep in self.multi_genre_separators:
            parts = [piece for part in parts for piece in part.split(sep)]
        names = (" ".join(part.casefold().split()) for part in parts)
        result = frozenset(self.aliases.get(n, n) for n in names if n)
        self._cache[raw] = result
        return result

    def are_neighbors(self, a: str, b: str) -> bool:
        return any(a in group and b in group for group in self.neighbor_groups)


class ExtrasConfig(_Config):
    vocal_clash_score: float = 0.3
    vocal_ok_score: float = 1.0
    rating_max: float = 5.0
    arrangement_good_score: float = 1.0
    arrangement_mid_score: float = 0.7
    arrangement_short_score: float = 0.4
    arrangement_long_min_bars: int = 16
    arrangement_short_below_bars: int = 8
    no_data_score: float = 0.5


class TransitionConfig(_Config):
    long_blend_bars: int = 32
    long_blend_style_bars: int = 64  # Phase 4: when a style profile prefers long blends
    filter_sweep_bars: int = 16
    drop_swap_bars: int = 16
    cut_bars: int = 8
    echo_out_bars: int = 8
    allowed_lengths_bars: tuple[int, ...] = (8, 16, 32, 64)
    drop_swap_min_energy_jump: float = 3.0
    long_blend_min_section_bars: int = 16  # intro/outro needed for a long blend
    sparse_intro_min_bars: int = 16  # Phase 3 proxy for "b starts sparse" -> echo_out


class SearchConfig(_Config):
    min_match: float = 0.5  # below this, a track query finds nothing
    confident_match: float = 0.8
    min_gap: float = 0.1  # best must beat second best by this to be unambiguous
    substring_base: float = 0.9
    artist_share: float = 0.4  # weight of artist vs title in "Artist - Title" queries
    max_candidates: int = 10


class DisplayConfig(_Config):
    good_score: float = 80.0  # green in tables
    ok_score: float = 60.0  # yellow in tables; below is red
    max_warnings_shown: int = 10
    wide_table_min_width: int = 140  # narrower terminals get the compact suggest table


class CurveTemplate(_Config):
    """Energy targets (1-10) over normalized set position 0-1, linearly interpolated."""

    points: tuple[tuple[float, float], ...]
    cap: float | None = None  # never target above this
    dip_every_tracks: int | None = None  # short dip on every Nth track
    dip_depth: float = 0.0

    @model_validator(mode="after")
    def _check_points(self) -> CurveTemplate:
        positions = [p for p, _ in self.points]
        if not positions or positions[0] != 0.0 or positions[-1] != 1.0:
            raise ValueError("curve points must start at position 0 and end at position 1")
        if positions != sorted(positions):
            raise ValueError("curve positions must be in increasing order")
        return self


class SetConfig(_Config):
    beam_width: int = 20
    alternates_per_position: int = 2
    alternate_max_energy_miss: float = 2.0  # alternates stay this close to the curve
    # Only pairs within this tempo difference (including half/double time) are considered.
    tempo_window_pct: float = 8.0
    # Hard constraint: an artist may not reappear within this many tracks (0 disables).
    artist_gap_tracks: int = 4
    # Soft penalties, in points subtracted from a step's score.
    artist_repeat_window: int = 4
    artist_repeat_penalty: float = 15.0
    label_run_length: int = 3
    label_run_penalty: float = 5.0
    energy_boost_window: int = 6
    energy_boost_max: int = 2
    energy_boost_penalty: float = 10.0
    max_tempo_drift_bpm: float = 8.0  # Phase 4 style profiles override this
    tempo_drift_penalty: float = 20.0
    tempo_reversal_bpm: float = 3.0  # tempo drops bigger than this while energy is rising
    tempo_reversal_penalty: float = 10.0
    style_fit_points: float = 15.0  # Phase 4: style fit x this is added to each step
    # Half/double-time tracks are folded into the octave of the set's first track.
    octave_fold_ratio: float = 1.414
    default_track_seconds: float = 360.0  # when TotalTime is missing
    seconds_per_minute: float = 60.0
    beats_per_bar: int = 4
    curves: dict[str, CurveTemplate] = {
        "warm_up": CurveTemplate(points=((0.0, 3.0), (1.0, 5.0)), cap=6.0),
        "peak_time": CurveTemplate(
            points=((0.0, 6.0), (1.0, 9.0)), dip_every_tracks=5, dip_depth=1.5
        ),
        "closing": CurveTemplate(points=((0.0, 8.0), (0.75, 8.0), (1.0, 5.5))),
        "journey": CurveTemplate(points=((0.0, 3.0), (0.7, 9.0), (1.0, 5.0))),
    }


class ScoringConfig(_Config):
    weights: ComponentWeights = ComponentWeights()
    tempo: TempoConfig = TempoConfig()
    harmonic: HarmonicConfig = HarmonicConfig()
    energy: EnergyConfig = EnergyConfig()
    genre: GenreConfig = GenreConfig()
    extras: ExtrasConfig = ExtrasConfig()
    transition: TransitionConfig = TransitionConfig()
    search: SearchConfig = SearchConfig()
    display: DisplayConfig = DisplayConfig()
    sets: SetConfig = SetConfig()


DEFAULT_CONFIG = ScoringConfig()


def band_score(value: float, bands: Bands, beyond: float) -> float:
    for max_value, score in bands:
        if value <= max_value:
            return score
    return beyond

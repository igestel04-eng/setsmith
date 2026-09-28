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
    style_profile_blend: float = 0.5  # share of the style profile's genre weight
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
    long_blend_style_bars: int = 64  # when a style profile prefers long blends
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
    # Version labels tracklists add or drop freely; ignored when matching titles. Remix
    # names are kept: they identify a different recording.
    version_noise: tuple[str, ...] = (
        "original mix", "extended mix", "extended version", "extended", "radio edit",
        "radio mix", "club mix", "main mix", "album version", "full length", "original",
    )  # fmt: skip


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
    style_fit_points: float = 15.0  # style fit (0-1) x this is added to each step
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


class AnalysisConfig(_Config):
    """Phase 3 local audio analysis."""

    version: int = 1  # bump when analysis logic changes, so cached results are redone
    sample_rate: int = 22050  # librosa features
    essentia_sample_rate: int = 44100
    hop_length: int = 512
    min_duration_s: float = 30.0  # shorter files are skipped
    essentia_key_profile: str = "edma"  # or "bgate", "edmm", "temperley", ...
    # Krumhansl-Kessler key profiles (C major / C minor) for the librosa fallback detector.
    kk_major: tuple[float, ...] = (
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    )
    kk_minor: tuple[float, ...] = (
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    )

    # Energy 1-10: weighted z-scores of these features, ranked across the library.
    energy_features: dict[str, float] = {
        "loudness_db": 0.4,
        "onset_rate": 0.3,
        "spectral_centroid_hz": 0.15,
        "spectral_flux": 0.15,
    }
    energy_decimals: int = 1

    # Intro/outro from per-bar harmonic energy on the beat grid. Harmonic/percussive
    # separation removes kicks and hats, so drum-only intros read as low. A coarse
    # spectrogram limited to 4 kHz keeps this well under a second per track.
    structure_n_fft: int = 4096
    structure_hop: int = 2048
    structure_max_hz: float = 4000.0
    hpss_kernel: int = 9
    level_floor_db: float = -120.0
    min_bar_fraction: float = 0.5  # drop a trailing partial bar shorter than this
    full_level_percentile: float = 75.0  # "full" level of the track
    full_margin_db: float = 4.0  # a bar within this of the full level counts as full
    full_min_bars: int = 4  # the full section must hold this long
    phrase_bars: int = 8  # intro/outro lengths snap to multiples of this
    max_section_fraction: float = 0.5  # longer "intros" are treated as detection failures
    default_beats_per_bar: int = 4

    # Key confidence after comparing the detected key with the Rekordbox tag.
    key_confidence_agree: float = 1.0
    key_confidence_related: float = 0.7  # adjacent or relative: common detector confusions
    key_confidence_disagree: float = 0.4
    key_confidence_detected_only: float = 0.8  # no usable tag: the detected key is used


class StyleConfig(_Config):
    """How a DJ style profile shapes scoring (Phase 4)."""

    key_move_blend: float = 0.5  # share of the profile's allowed_key_moves weight
    unlisted_genre_weight: float = 0.2  # genre the profile does not mention
    unknown_genre_weight: float = 0.5  # track without a genre
    # Style fit = weighted average of the parts that are known for the track.
    fit_weights: dict[str, float] = {"bpm": 0.35, "genre": 0.35, "key_mode": 0.2, "vocal": 0.1}
    bpm_falloff_bpm: float = 4.0  # BPM fit falls from 1 at the band edge to 0 this far out
    long_blend_64_min_share: float = 0.25  # profile "prefers long blends" at this 64-bar share
    mix_tolerance: float = 0.02  # transition mixes must sum to 1 within this


class VocalTagConfig(_Config):
    """Vocal / instrumental from metadata, until audio vocal detection exists.

    Whole words in the title, Mix field or Comments. Both kinds present means unknown.
    """

    vocal_words: tuple[str, ...] = ("vocal", "vocals", "vox", "acapella", "a cappella")
    instrumental_words: tuple[str, ...] = ("instrumental", "inst", "dub", "no vocals")


class RekordboxDbConfig(_Config):
    keep_copies: int = 3  # master.db copies kept in Setsmith's folder; older ones are pruned


class LiveSetConfig(_Config):
    """Phase 5: analysis of recorded sets and tracklists."""

    window_s: float = 30.0  # windowed tempo/key/loudness on the recording
    sample_rate: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024  # ~46 ms frames for alignment features
    n_mfcc: int = 13
    mfcc_weight: float = 0.5  # MFCC vs chroma share in the alignment distance
    # Subsequence DTW of each original track against the mix (Kim et al. 2020).
    search_margin_s: float = 180.0  # around tracklist timestamps
    max_search_s: float = 1200.0  # mix span searched per track without timestamps
    # Cost multiplier for DTW steps that stall one side. Above 1 keeps the path on the 1:1
    # diagonal instead of sliding between repeated loops (common in dance music).
    dtw_offdiagonal_weight: float = 2.0
    diagonal_window_beats: int = 16  # sliding window for "playing at normal speed"
    diagonal_min_share: float = 0.75  # share of 1:1 beat steps inside that window
    min_played_beats: int = 32  # shorter matches are treated as not found
    max_match_cost: float = 0.35  # mean cosine distance above this = not found
    novelty_half_window_bars: int = 16  # boundary detection without timestamps or DTW
    # After DTW fixes each track's offset, per-beat gains (non-negative least squares on
    # mel power: mix ~ sum of gain x track) show where each track is audible, including
    # fades that similarity alone cannot see.
    # A fader moves amplitude (sqrt of power gain) roughly linearly, so each span edge is
    # found by fitting a line to the fade and extrapolating it to silence.
    n_mels: int = 64
    amp_smooth_beats: int = 4
    fade_fit_range: tuple[float, float] = (0.2, 0.85)  # relative amplitudes used for the fit
    fade_floor: float = 0.15  # stop walking outwards below this relative amplitude
    fade_rise_tolerance: float = 0.1  # ...or when amplitude rises again by this much
    fade_min_points: int = 4
    max_extend_beats: int = 256  # how far a span may grow beyond its DTW core
    # Estimating transition types from measured overlaps.
    blend_min_bars: float = 16.0  # overlap at least this long counts as a long blend
    sweep_min_bars: float = 4.0  # shorter overlaps count as cuts
    min_overlap_bars: float = 4.0  # shorter overlaps are not snapped to a blend length
    # Draft profiles.
    bpm_band_percentiles: tuple[float, float] = (10.0, 90.0)
    bpm_round: float = 0.5
    move_smoothing: float = 1.0  # add-one smoothing for key-move weights
    energy_smooth_windows: int = 3
    curve_points: int = 5
    top_references: int = 5
    # Learned preferences: learned values get weight n / (n + learn_prior).
    learn_prior: float = 50.0
    default_length_mix: dict[str, float] = {"8": 0.1, "16": 0.3, "32": 0.45, "64": 0.15}
    default_type_mix: dict[str, float] = {
        "long_blend": 0.5, "cut": 0.1, "echo_out": 0.1, "filter_sweep": 0.2, "drop_swap": 0.1,
    }  # fmt: skip


class DiscoveryConfig(_Config):
    """Finding new tracks through official APIs with the user's own keys."""

    lastfm_base: str = "https://ws.audioscrobbler.com/2.0/"
    getsongbpm_base: str = "https://api.getsong.co"
    similar_tracks: int = 50  # Last.fm track.getSimilar results per seed
    similar_artists: int = 10  # similar artists whose top tracks become candidates
    top_tracks_per_artist: int = 5
    # Scene match: how similar a candidate's artist is to the seed's artist on Last.fm.
    # "Listeners also played" track similarity drifts to mainstream hits for crossover
    # seeds; artist similarity stays in the seed's scene. This many similar artists are
    # fetched (in the same request as `similar_artists`) to measure it.
    scene_artists: int = 100
    # Candidates below this scene match (or whose artist is not among `scene_artists`)
    # are "different scene" and listed last.
    min_scene_match: float = 0.2
    max_lookups: int = 25  # BPM/key lookups per discovery (most relevant candidates first)
    max_per_artist: int = 2  # keep results varied
    # Results with known BPM and key at or above this score rank first; unknowns follow
    # (by relevance: Last.fm similarity and scene match); known poor matches come last.
    good_match_score: float = 60.0
    # BPM and key estimated from Deezer's official 30-second previews, for tracks GetSongBPM
    # doesn't know. Decoded in memory and discarded; only the numbers are kept.
    deezer_base: str = "https://api.deezer.com"
    deezer_search_results: int = 5  # search hits checked for the right artist and title
    preview_host_suffix: str = ".dzcdn.net"  # previews are only fetched from Deezer's CDN
    preview_max_bytes: int = 2_000_000  # a 30 s MP3 is about 0.5 MB
    preview_min_s: float = 10.0  # shorter previews are not analyzed
    preview_workers: int = 4  # previews analyzed in parallel
    preview_start_bpm: float = 120.0  # beat tracker prior (house tempo)
    preview_bpm_fold_min: float = 88.0  # tempos are halved/doubled into [min, 2 x min)
    preview_min_beats: int = 8  # beats needed to refine the tempo from beat positions
    # With Essentia installed, three tempo estimators vote: the BPM is the mean of those
    # that agree within this; when none do, no BPM is given (a wrong BPM is worse than none).
    preview_bpm_agree_pct: float = 2.0
    preview_bpm_decimals: int = 1
    # 30 seconds of audio gives a rough key: below the harmonic low-confidence flag, so
    # key scores are pulled toward neutral.
    preview_key_confidence: float = 0.6
    preview_cache_days: float = 180.0  # derived BPM/key don't change
    # Deezer search results are kept briefly: the preview links in them expire after about
    # 15 minutes.
    deezer_cache_minutes: float = 10.0
    # Polite pacing: GetSongBPM allows 3,000 requests an hour; Last.fm asks for restraint;
    # Deezer allows 50 requests per 5 seconds.
    lastfm_min_interval_s: float = 0.25
    getsongbpm_min_interval_s: float = 0.6
    deezer_min_interval_s: float = 0.15
    timeout_s: float = 15.0
    cache_ttl_days: float = 7.0
    cache_max_entries: int = 20000  # keeps the cache far below Last.fm's 100 MB storage cap


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
    analysis: AnalysisConfig = AnalysisConfig()
    style: StyleConfig = StyleConfig()
    vocal: VocalTagConfig = VocalTagConfig()
    rekordbox_db: RekordboxDbConfig = RekordboxDbConfig()
    liveset: LiveSetConfig = LiveSetConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()


DEFAULT_CONFIG = ScoringConfig()


def band_score(value: float, bands: Bands, beyond: float) -> float:
    for max_value, score in bands:
        if value <= max_value:
            return score
    return beyond

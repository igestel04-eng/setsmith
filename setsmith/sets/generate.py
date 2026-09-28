"""Energy-curve set generation with beam search.

Each step from track a to b at position i scores:

    base(a -> b)                      transition score without the energy component
  + energy component x weight x 100   with target change = curve[i] - energy(a)
  + style fit x 15                    Phase 4 (0 until style profiles exist)
  - penalties                         artist repeats, label runs, stacked energy boosts,
                                      tempo drift, tempo reversals while energy rises

The beam keeps the best `beam_width` partial sets by summed step score.
"""

from __future__ import annotations

import heapq
import math
import re
import statistics
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from setsmith.graph.build import CompatibilityGraph, PairKey, PairScore
from setsmith.io.enrich import tag_names
from setsmith.model.collection import Collection, normalize_text, strip_versions
from setsmith.model.track import Track
from setsmith.scoring.transition import TransitionScore, energy_score, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig, band_score
from setsmith.sets.curves import EnergyCurve

if TYPE_CHECKING:
    from setsmith.styles.profile import StyleProfile

# Collaboration separators. "and"/"with" are left out: they occur inside names.
_ARTIST_SPLIT_RE = re.compile(r"\s*(?:,|&|\s(?:x|feat\.?|ft\.?|vs\.?)\s)\s*")


@lru_cache(maxsize=65536)
def artist_names(artist: str) -> frozenset[str]:
    """Individual artists in an Artist tag: "&ME, Rampa & Adam Port" -> 3 names."""
    # NFKC turns full-width separators (U+FF0C, common in SoundCloud uploads) into plain ones.
    parts = _ARTIST_SPLIT_RE.split(unicodedata.normalize("NFKC", artist).casefold())
    return frozenset(n for p in parts if (n := normalize_text(p)))


@dataclass(frozen=True)
class SetRequest:
    curve: EnergyCurve
    minutes: float | None = None
    track_count: int | None = None
    start_id: str | None = None
    bpm_min: float | None = None
    bpm_max: float | None = None
    genres: frozenset[str] = frozenset()  # normalized; empty = any
    exclude_ids: frozenset[str] = frozenset()
    artist_gap: int | None = None  # None = config default
    max_tempo_drift_bpm: float | None = None  # None = config default
    beam_width: int | None = None  # None = config default
    style: StyleProfile | None = None  # BPM band, drift, key/genre weights, style fit
    tags: frozenset[str] = frozenset()  # lowercased My Tag names; empty = any
    # Songs outside the library mixed into the pool: about this share of the set's tracks
    # comes from `new_ids`, never more (a fixed opening track doesn't count). Until the
    # share is reached, new songs get a bonus in the search: their keys are estimates.
    new_ids: frozenset[str] = frozenset()
    max_new_share: float = 1.0

    @property
    def bpm_range(self) -> tuple[float | None, float | None]:
        """Explicit limits win; otherwise the style's BPM band is a hard limit."""
        band = self.style.bpm_band if self.style else (None, None)
        return (
            self.bpm_min if self.bpm_min is not None else band[0],
            self.bpm_max if self.bpm_max is not None else band[1],
        )


@dataclass(frozen=True, slots=True)
class Alternate:
    track: Track
    fit: float  # combined step points in and out, comparable across alternates only
    into: TransitionScore | None  # from the previous track
    out_of: TransitionScore | None  # into the next track


@dataclass(slots=True)
class SetPosition:
    index: int  # 0-based
    track: Track
    target_energy: float
    transition: TransitionScore | None  # into the next track, None for the last
    alternates: list[Alternate] = field(default_factory=list)


@dataclass(slots=True)
class SetStats:
    track_count: int
    estimated_seconds: float
    mean_transition: float
    min_transition: float
    curve_mae: float | None  # mean |energy - target| over tracks with known energy
    bpm_min: float
    bpm_max: float
    pool_size: int
    pairs_scored: int
    pairs_from_cache: int
    style: str | None = None
    mean_style_fit: float | None = None
    transition_mix: dict[str, float] = field(default_factory=dict)  # share per type
    style_transition_mix: dict[str, float] = field(default_factory=dict)  # the style's target

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "mean_style_fit": None
            if self.mean_style_fit is None
            else round(self.mean_style_fit, 2),
            "transition_mix": {k: round(v, 2) for k, v in self.transition_mix.items()},
            "style_transition_mix": self.style_transition_mix,
            "track_count": self.track_count,
            "estimated_minutes": round(self.estimated_seconds / 60, 1),
            "mean_transition": round(self.mean_transition, 1),
            "min_transition": round(self.min_transition, 1),
            "curve_mae": None if self.curve_mae is None else round(self.curve_mae, 2),
            "bpm_min": self.bpm_min,
            "bpm_max": self.bpm_max,
            "pool_size": self.pool_size,
            "pairs_scored": self.pairs_scored,
            "pairs_from_cache": self.pairs_from_cache,
        }


@dataclass(slots=True)
class GeneratedSet:
    curve: str
    positions: list[SetPosition]
    stats: SetStats
    warnings: list[str]
    # Songs outside the library, by track ID: where to get them (links) and how they
    # were found. Filled in by the caller that added them to the pool.
    outside: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def track_ids(self) -> list[str]:
        return [p.track.id for p in self.positions]

    @property
    def alternate_ids(self) -> list[str]:
        seen = set(self.track_ids)
        out: list[str] = []
        for p in self.positions:
            for alt in p.alternates:
                if alt.track.id not in seen:
                    seen.add(alt.track.id)
                    out.append(alt.track.id)
        return out


class SetGenerationError(ValueError):
    pass


# ---------------------------------------------------------------- pool


def filter_pool(
    collection: Collection, request: SetRequest, cfg: ScoringConfig = DEFAULT_CONFIG
) -> tuple[list[Track], list[str]]:
    """Tracks eligible for the set, plus warnings about what was left out and why."""
    pool: list[Track] = []
    no_bpm = 0
    bpm_min, bpm_max = request.bpm_range
    for t in collection.tracks.values():
        if t.id in request.exclude_ids:
            continue
        if not t.bpm:
            no_bpm += 1
            continue
        if bpm_min is not None and t.bpm < bpm_min:
            continue
        if bpm_max is not None and t.bpm > bpm_max:
            continue
        if request.genres and not (cfg.genre.normalize(t.genre) & request.genres):
            continue
        if request.tags and not (tag_names(t) & request.tags):
            continue
        pool.append(t)

    warnings = []
    if no_bpm:
        warnings.append(f"{no_bpm} track(s) without BPM left out (analyze them in Rekordbox)")
    if request.start_id is not None:
        start = collection.tracks.get(request.start_id)
        if start is None:
            raise SetGenerationError(f"start track {request.start_id!r} not in collection")
        if not start.bpm:
            raise SetGenerationError(f"start track {start.display!r} has no BPM")
        if start not in pool:
            pool.append(start)
            warnings.append(f"start track {start.display!r} is outside the filters; kept anyway")
    return pool, warnings


def estimate_track_count(
    pool: list[Track], minutes: float, cfg: ScoringConfig, style: StyleProfile | None = None
) -> int:
    """Tracks needed for `minutes`: median length minus a long-blend overlap per track."""
    sc = cfg.sets
    long_style = style is not None and style.prefers_long_blends(cfg)
    blend_bars = (
        cfg.transition.long_blend_style_bars if long_style else cfg.transition.long_blend_bars
    )
    durations = [t.duration_s or sc.default_track_seconds for t in pool]
    bpms = [t.bpm for t in pool if t.bpm]
    median_len = statistics.median(durations) if durations else sc.default_track_seconds
    overlap = bars_to_seconds(blend_bars, statistics.median(bpms), cfg) if bpms else 0.0
    per_track = max(median_len - overlap, overlap, 1.0)
    return max(2, round(minutes * sc.seconds_per_minute / per_track))


def bars_to_seconds(bars: int, bpm: float, cfg: ScoringConfig) -> float:
    return bars * cfg.sets.beats_per_bar * cfg.sets.seconds_per_minute / bpm


# ---------------------------------------------------------------- search


@dataclass(frozen=True, slots=True)
class _State:
    path: tuple[str, ...]
    score: float
    boosts: tuple[bool, ...]  # energy-boost key move, per transition so far
    ref_bpm: float  # first track's BPM: half/double-time tracks fold into its octave
    bpm_lo: float  # folded tempo range so far
    bpm_hi: float


class _Search:
    def __init__(
        self,
        graph: CompatibilityGraph,
        targets: list[float],
        request: SetRequest,
        cfg: ScoringConfig,
    ) -> None:
        sc = cfg.sets
        self.g = graph
        self.t = graph.tracks
        self.targets = targets
        self.cfg = cfg
        self.width = request.beam_width or sc.beam_width
        self.artist_gap = sc.artist_gap_tracks if request.artist_gap is None else request.artist_gap
        self.style = request.style
        if request.max_tempo_drift_bpm is not None:
            self.max_drift = request.max_tempo_drift_bpm
        elif self.style is not None:
            self.max_drift = self.style.max_tempo_drift_bpm
        else:
            self.max_drift = sc.max_tempo_drift_bpm
        self.new_ids = request.new_ids
        self.max_new: int | None = None  # set once the track count is known
        self.fixed_start = request.start_id is not None
        self.energy_weight_pts = 100 * cfg.weights.energy
        self.style_max_pts = sc.style_fit_points if self.style else 0.0
        self._style_pts: dict[str, float] = {}
        # Duplicate collection entries of one song share a key, so a set never repeats it.
        self.song = {
            tid: (normalize_text(t.artist), strip_versions(normalize_text(t.title)))
            if t.title
            else (tid, "")
            for tid, t in graph.tracks.items()
        }

    def used_songs(self, path: Iterable[str]) -> set[tuple[str, str]]:
        return {self.song[p] for p in path}

    # -- scoring helpers

    def fold(self, bpm: float | None, ref: float) -> float:
        ratio = self.cfg.sets.octave_fold_ratio
        value = bpm or ref
        while value > ref * ratio:
            value /= 2
        while value < ref / ratio:
            value *= 2
        return value

    def target_delta(self, a: Track, i: int) -> float:
        """Energy change wanted from a to the track at position i: land on the curve."""
        if a.energy is not None:
            return self.targets[i] - a.energy
        return self.targets[i] - self.targets[i - 1]

    def style_pts(self, t: Track) -> float:
        """Style fit x style_fit_points (0 without a style)."""
        if self.style is None:
            return 0.0
        pts = self._style_pts.get(t.id)
        if pts is None:
            pts = self.cfg.sets.style_fit_points * self.style.style_fit(t, self.cfg).score
            self._style_pts[t.id] = pts
        return pts

    def energy_pts(self, a: Track, b: Track, i: int) -> float:
        return self.energy_weight_pts * energy_score(a, b, self.target_delta(a, i), self.cfg)

    def start_pts(self, t: Track) -> float:
        e = self.cfg.energy
        if t.energy is None:
            fit = e.missing_score
        else:
            fit = band_score(abs(t.energy - self.targets[0]), e.bands, e.beyond_score)
        return self.energy_weight_pts * fit + self.style_pts(t)

    def step(self, a_id: str, b_id: str, i: int) -> float:
        """Path-independent step points (base + energy + style), used for alternates."""
        base, _ = self.g.pair(a_id, b_id)
        b = self.t[b_id]
        return base + self.energy_pts(self.t[a_id], b, i) + self.style_pts(b)

    def shares_artist(self, b: Track, others: Iterable[str]) -> bool:
        names = artist_names(b.artist)
        return bool(names) and any(names & artist_names(self.t[o].artist) for o in others)

    def penalties(self, state: _State, b: Track, i: int, boost: bool) -> tuple[float, float, float]:
        """(penalty points, new bpm_lo, new bpm_hi) for appending b at position i."""
        sc = self.cfg.sets
        path = state.path
        penalty = 0.0

        if self.shares_artist(b, path[-sc.artist_repeat_window :]):
            penalty += sc.artist_repeat_penalty

        run = sc.label_run_length - 1
        if b.label and len(path) >= run and all(self.t[p].label == b.label for p in path[-run:]):
            penalty += sc.label_run_penalty

        window = (*state.boosts[-(sc.energy_boost_window - 1) :], boost)
        if boost and sum(window) > sc.energy_boost_max:
            penalty += sc.energy_boost_penalty

        fb = self.fold(b.bpm, state.ref_bpm)
        lo, hi = min(state.bpm_lo, fb), max(state.bpm_hi, fb)
        if hi - lo > self.max_drift and hi - lo > state.bpm_hi - state.bpm_lo:
            penalty += sc.tempo_drift_penalty

        rising = self.targets[i] > self.targets[i - 1]
        fa = self.fold(self.t[path[-1]].bpm, state.ref_bpm)
        if rising and fb < fa - sc.tempo_reversal_bpm:
            penalty += sc.tempo_reversal_penalty
        return penalty, lo, hi

    # -- beam search

    def initial_states(self, start_id: str | None) -> list[_State]:
        if start_id is not None:
            starts = [self.t[start_id]]
        else:
            preferred = self.style.bpm_preferred if self.style else None

            def rank(t: Track) -> tuple[float, float, int, str, str]:
                off_tempo = abs((t.bpm or 0.0) - preferred) if preferred else 0.0
                return (-self.start_pts(t), off_tempo, -t.rating, t.display, t.id)

            ranked = sorted(self.t.values(), key=rank)
            if self.max_new == 0:
                ranked = [t for t in ranked if t.id not in self.new_ids]
            starts = ranked[: self.width]
        states = []
        for track in starts:
            ref = track.bpm or 0.0
            states.append(_State((track.id,), self.start_pts(track), (), ref, ref, ref))
        return states

    def extend(self, state: _State, i: int) -> list[_State]:
        """The best `width` one-track extensions of a state."""
        a_id = state.path[-1]
        a = self.t[a_id]
        used = self.used_songs(state.path)
        blocked = state.path[-self.artist_gap :] if self.artist_gap > 0 else ()
        counted = state.path[1:] if self.fixed_start else state.path
        new_count = sum(1 for p in counted if p in self.new_ids)
        new_full = self.max_new is not None and new_count >= self.max_new
        # Paced toward the share so new songs spread through the set: a bonus while the set
        # is behind at this point, and only new songs once it is a whole song behind (their
        # estimated keys and missing energy would otherwise keep them out).
        share_here = (self.max_new or 0) * (i + 1) / len(self.targets)
        new_bonus = (
            self.cfg.sets.new_song_bonus_pts
            if new_count < math.ceil(share_here) and not new_full
            else 0.0
        )
        only_new = new_count < math.floor(share_here) and not new_full

        def scan(only_new: bool) -> list[_State]:
            best: list[tuple[float, int, _State]] = []
            for n, (b_id, ceiling) in enumerate(self.g.candidates(a_id)):
                # Candidates come sorted by a ceiling on base; energy and style add at most
                # their maximum points and penalties only subtract, so once the ceiling
                # loses, every later candidate does too.
                bound = (
                    state.score + ceiling + self.energy_weight_pts + self.style_max_pts + new_bonus
                )
                if len(best) >= self.width and bound <= best[0][0]:
                    break
                is_new = b_id in self.new_ids
                if self.song[b_id] in used or (new_full and is_new) or (only_new and not is_new):
                    continue
                b = self.t[b_id]
                if blocked and self.shares_artist(b, blocked):
                    continue
                base, boost = self.g.pair(a_id, b_id)
                penalty, lo, hi = self.penalties(state, b, i, boost)
                score = state.score + base + self.energy_pts(a, b, i) + self.style_pts(b) - penalty
                if is_new:
                    score += new_bonus
                path = (*state.path, b_id)
                new = _State(path, score, (*state.boosts, boost), state.ref_bpm, lo, hi)
                if len(best) < self.width:
                    heapq.heappush(best, (score, n, new))
                elif score > best[0][0]:
                    heapq.heapreplace(best, (score, n, new))
            return [s for _, _, s in best]

        # No new song fits here (tempo, artist gap): fall back to the whole pool.
        return (only_new and scan(True)) or scan(False)

    def run(self, n: int, start_id: str | None) -> list[str]:
        """Best path of up to n tracks (shorter if the pool runs dry)."""
        beam = self.initial_states(start_id)
        for i in range(1, n):
            candidates = [new for state in beam for new in self.extend(state, i)]
            if not candidates:
                break
            candidates.sort(key=lambda s: (-s.score, s.path))
            beam = candidates[: self.width]
        return list(max(beam, key=lambda s: s.score).path)


# ---------------------------------------------------------------- public API


def generate_set(
    collection: Collection,
    request: SetRequest,
    *,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    cache: dict[PairKey, PairScore] | None = None,
) -> tuple[GeneratedSet, CompatibilityGraph]:
    """Build a set that follows the request's energy curve.

    Returns the set and the graph used, whose `new_scores` a caller may persist.
    """
    pool, warnings = filter_pool(collection, request, cfg)
    if len(pool) < 2:  # noqa: PLR2004 - a set needs at least one transition
        raise SetGenerationError(f"only {len(pool)} track(s) match the filters")

    if request.track_count is not None:
        n = request.track_count
    elif request.minutes is not None:
        n = estimate_track_count(pool, request.minutes, cfg, request.style)
    else:
        raise SetGenerationError("give a set length in minutes or a track count")
    if n > len(pool):
        warnings.append(f"asked for {n} tracks but only {len(pool)} match; using all of them")
        n = len(pool)

    graph = CompatibilityGraph(pool, cfg, cache, request.style)
    targets = request.curve.targets(n)
    search = _Search(graph, targets, request, cfg)
    if request.new_ids:
        search.max_new = round(request.max_new_share * n)
    path = search.run(n, request.start_id)
    reached = len(path)
    if reached < n:
        warnings.append(
            f"ran out of compatible tracks after {reached} of {n}; "
            "widen the BPM range or genres, or lower --artist-gap"
        )
        targets = request.curve.targets(reached)
        search.targets = targets

    positions = _positions(search, path, targets, request)
    stats = _stats(positions, graph, len(pool), len(cache or {}), cfg, request.style)
    return GeneratedSet(request.curve.name, positions, stats, warnings), graph


def _positions(
    search: _Search, path: list[str], targets: list[float], request: SetRequest
) -> list[SetPosition]:
    t = search.t
    positions = []
    for i, tid in enumerate(path):
        transition = None
        if i + 1 < len(path):
            a = t[tid]
            transition = score_transition(
                a,
                t[path[i + 1]],
                energy_target=search.target_delta(a, i + 1),
                cfg=search.cfg,
                style=search.style,
            )
        positions.append(SetPosition(i, t[tid], targets[i], transition))
    fixed_start = request.start_id is not None
    for pos in positions:
        if pos.index == 0 and fixed_start:
            continue
        pos.alternates = _alternates(search, path, pos.index)
    return positions


def _alternates(search: _Search, path: list[str], i: int) -> list[Alternate]:
    """Tracks that could replace path[i], fitting both the previous and the next track."""
    count = search.cfg.sets.alternates_per_position
    max_miss = search.cfg.sets.alternate_max_energy_miss
    if count <= 0:
        return []
    t, g = search.t, search.g
    used = search.used_songs(path)
    prev_id = path[i - 1] if i > 0 else None
    next_id = path[i + 1] if i + 1 < len(path) else None
    gap = search.artist_gap
    neighbors = path[max(0, i - gap) : i] + path[i + 1 : i + 1 + gap] if gap > 0 else []

    # Ceilings on the step into each candidate, highest first.
    ceilings: list[tuple[str, float]]
    if prev_id is None:
        ceilings = sorted(((c, search.start_pts(t[c])) for c in t), key=lambda cf: -cf[1])
    else:
        e_max = search.energy_weight_pts + search.style_max_pts
        ceilings = [(c, ub + e_max) for c, ub in g.candidates(prev_id)]

    best: list[tuple[float, str]] = []
    max_second = 100.0 + search.style_max_pts if next_id else 0.0  # a step's ceiling
    for c_id, ceiling in ceilings:
        if len(best) >= count and ceiling + max_second <= best[0][0]:
            break
        if search.song[c_id] in used:
            continue
        energy = t[c_id].energy
        if energy is not None and abs(energy - search.targets[i]) > max_miss:
            continue
        if neighbors and search.shares_artist(t[c_id], neighbors):
            continue
        if next_id is not None and not g.in_window(c_id, next_id):
            continue
        first = search.start_pts(t[c_id]) if prev_id is None else search.step(prev_id, c_id, i)
        fit = first + (search.step(c_id, next_id, i + 1) if next_id else 0.0)
        if len(best) < count:
            heapq.heappush(best, (fit, c_id))
        elif fit > best[0][0]:
            heapq.heapreplace(best, (fit, c_id))

    cfg = search.cfg
    out = []
    for fit, c_id in sorted(best, reverse=True):
        c = t[c_id]
        into = out_of = None
        if prev_id is not None:
            a = t[prev_id]
            into = score_transition(
                a, c, energy_target=search.target_delta(a, i), cfg=cfg, style=search.style
            )
        if next_id is not None:
            delta = search.target_delta(c, i + 1)
            out_of = score_transition(
                c, t[next_id], energy_target=delta, cfg=cfg, style=search.style
            )
        out.append(Alternate(c, fit, into, out_of))
    return out


def _stats(
    positions: list[SetPosition],
    graph: CompatibilityGraph,
    pool_size: int,
    cached: int,
    cfg: ScoringConfig,
    style: StyleProfile | None = None,
) -> SetStats:
    totals = [p.transition.total for p in positions if p.transition]
    seconds = 0.0
    for p in positions:
        seconds += p.track.duration_s or cfg.sets.default_track_seconds
        if p.transition and p.track.bpm:
            seconds -= bars_to_seconds(p.transition.suggested_length_bars, p.track.bpm, cfg)
    misses = [
        abs(p.track.energy - p.target_energy) for p in positions if p.track.energy is not None
    ]
    bpms = [p.track.bpm for p in positions if p.track.bpm]
    kinds = [p.transition.suggested_type.value for p in positions if p.transition]
    mix = {k: kinds.count(k) / len(kinds) for k in sorted(set(kinds))} if kinds else {}
    fits = [style.style_fit(p.track, cfg).score for p in positions] if style else []
    return SetStats(
        track_count=len(positions),
        estimated_seconds=seconds,
        mean_transition=statistics.fmean(totals) if totals else 0.0,
        min_transition=min(totals) if totals else 0.0,
        curve_mae=statistics.fmean(misses) if misses else None,
        bpm_min=min(bpms) if bpms else 0.0,
        bpm_max=max(bpms) if bpms else 0.0,
        pool_size=pool_size,
        pairs_scored=len(graph.new_scores),
        pairs_from_cache=cached,
        style=style.name if style else None,
        mean_style_fit=statistics.fmean(fits) if fits else None,
        transition_mix=mix,
        style_transition_mix={str(k): v for k, v in style.transition_type_mix.items()}
        if style
        else {},
    )

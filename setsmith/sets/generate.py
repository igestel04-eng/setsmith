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
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from setsmith.graph.build import CompatibilityGraph, PairKey, PairScore
from setsmith.model.collection import Collection, normalize_text
from setsmith.model.track import Track
from setsmith.scoring.transition import TransitionScore, energy_score, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig, band_score
from setsmith.sets.curves import EnergyCurve

# Collaboration separators. "and"/"with" are left out: they occur inside names.
_ARTIST_SPLIT_RE = re.compile(r"\s*(?:,|&|\s(?:x|feat\.?|ft\.?|vs\.?)\s)\s*")


@lru_cache(maxsize=65536)
def artist_names(artist: str) -> frozenset[str]:
    """Individual artists in an Artist tag: "&ME, Rampa & Adam Port" -> 3 names."""
    parts = _ARTIST_SPLIT_RE.split(artist.casefold())
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    for t in collection.tracks.values():
        if t.id in request.exclude_ids:
            continue
        if not t.bpm:
            no_bpm += 1
            continue
        if request.bpm_min is not None and t.bpm < request.bpm_min:
            continue
        if request.bpm_max is not None and t.bpm > request.bpm_max:
            continue
        if request.genres and not (cfg.genre.normalize(t.genre) & request.genres):
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


def estimate_track_count(pool: list[Track], minutes: float, cfg: ScoringConfig) -> int:
    """Tracks needed for `minutes`: median length minus a long-blend overlap per track."""
    sc = cfg.sets
    durations = [t.duration_s or sc.default_track_seconds for t in pool]
    bpms = [t.bpm for t in pool if t.bpm]
    median_len = statistics.median(durations) if durations else sc.default_track_seconds
    overlap = (
        _bars_to_seconds(cfg.transition.long_blend_bars, statistics.median(bpms), cfg)
        if bpms
        else 0.0
    )
    per_track = max(median_len - overlap, overlap, 1.0)
    return max(2, round(minutes * sc.seconds_per_minute / per_track))


def _bars_to_seconds(bars: int, bpm: float, cfg: ScoringConfig) -> float:
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
        self.max_drift = (
            sc.max_tempo_drift_bpm
            if request.max_tempo_drift_bpm is None
            else request.max_tempo_drift_bpm
        )
        self.energy_weight_pts = 100 * cfg.weights.energy
        # Duplicate collection entries of one song share a key, so a set never repeats it.
        self.song = {
            tid: (normalize_text(t.artist), normalize_text(t.title)) if t.title else (tid, "")
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

    def energy_pts(self, a: Track, b: Track, i: int) -> float:
        return self.energy_weight_pts * energy_score(a, b, self.target_delta(a, i), self.cfg)

    def start_pts(self, t: Track) -> float:
        e = self.cfg.energy
        if t.energy is None:
            fit = e.missing_score
        else:
            fit = band_score(abs(t.energy - self.targets[0]), e.bands, e.beyond_score)
        return self.energy_weight_pts * fit

    def step(self, a_id: str, b_id: str, i: int) -> float:
        """Path-independent step points (base + energy), used for alternates."""
        base, _ = self.g.pair(a_id, b_id)
        return base + self.energy_pts(self.t[a_id], self.t[b_id], i)

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
            ranked = sorted(
                self.t.values(), key=lambda t: (-self.start_pts(t), -t.rating, t.display, t.id)
            )
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
        best: list[tuple[float, int, _State]] = []
        for n, (b_id, ceiling) in enumerate(self.g.candidates(a_id)):
            # Candidates come sorted by a ceiling on base; energy adds at most
            # energy_weight_pts and penalties only subtract, so once the ceiling loses,
            # every later candidate does too.
            if (
                len(best) >= self.width
                and state.score + ceiling + self.energy_weight_pts <= (best[0][0])
            ):
                break
            if self.song[b_id] in used:
                continue
            b = self.t[b_id]
            if blocked and self.shares_artist(b, blocked):
                continue
            base, boost = self.g.pair(a_id, b_id)
            penalty, lo, hi = self.penalties(state, b, i, boost)
            score = state.score + base + self.energy_pts(a, b, i) - penalty
            new = _State((*state.path, b_id), score, (*state.boosts, boost), state.ref_bpm, lo, hi)
            if len(best) < self.width:
                heapq.heappush(best, (score, n, new))
            elif score > best[0][0]:
                heapq.heapreplace(best, (score, n, new))
        return [s for _, _, s in best]

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
        n = estimate_track_count(pool, request.minutes, cfg)
    else:
        raise SetGenerationError("give a set length in minutes or a track count")
    if n > len(pool):
        warnings.append(f"asked for {n} tracks but only {len(pool)} match; using all of them")
        n = len(pool)

    graph = CompatibilityGraph(pool, cfg, cache)
    targets = request.curve.targets(n)
    search = _Search(graph, targets, request, cfg)
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
    stats = _stats(positions, graph, len(pool), len(cache or {}), cfg)
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
                a, t[path[i + 1]], energy_target=search.target_delta(a, i + 1), cfg=search.cfg
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
        e_max = search.energy_weight_pts
        ceilings = [(c, ub + e_max) for c, ub in g.candidates(prev_id)]

    best: list[tuple[float, str]] = []
    max_second = 100.0 if next_id else 0.0  # a step never scores above 100
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
            into = score_transition(a, c, energy_target=search.target_delta(a, i), cfg=cfg)
        if next_id is not None:
            delta = search.target_delta(c, i + 1)
            out_of = score_transition(c, t[next_id], energy_target=delta, cfg=cfg)
        out.append(Alternate(c, fit, into, out_of))
    return out


def _stats(
    positions: list[SetPosition],
    graph: CompatibilityGraph,
    pool_size: int,
    cached: int,
    cfg: ScoringConfig,
) -> SetStats:
    totals = [p.transition.total for p in positions if p.transition]
    seconds = 0.0
    for p in positions:
        seconds += p.track.duration_s or cfg.sets.default_track_seconds
        if p.transition and p.track.bpm:
            seconds -= _bars_to_seconds(p.transition.suggested_length_bars, p.track.bpm, cfg)
    misses = [
        abs(p.track.energy - p.target_energy) for p in positions if p.track.energy is not None
    ]
    bpms = [p.track.bpm for p in positions if p.track.bpm]
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
    )

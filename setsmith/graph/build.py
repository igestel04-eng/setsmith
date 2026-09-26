"""Compatibility graph over a pool of tracks.

Edges run from a to every b within the tempo window (including half/double time). Each edge
carries the energy-independent part of the transition score ("base" points: total minus the
energy component) and whether the key move is an energy boost. The energy component depends
on where in the set the pair sits, so set generation adds it per position.

`candidates(a)` lists neighbors by a cheap upper bound on their base score, so searches can
stop before fully scoring pairs that cannot win. Full scores are computed on demand by
`pair()`, memoized, and can be persisted through `cache`.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from typing import Any

from setsmith.model.track import Track
from setsmith.scoring.transition import (
    SCORER_VERSION,
    Flag,
    base_upper_bound,
    score_transition,
)
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

PairKey = tuple[str, str]
PairScore = tuple[float, bool]  # (base points, energy-boost key move)

_TEMPO_MULTIPLIERS = (1.0, 2.0, 0.5)


def track_fingerprint(t: Track) -> str:
    """Hash of every field that affects scoring; changes when the track's tags change."""
    fields = (
        t.bpm, t.camelot, t.key_confidence, t.energy, t.genre, t.rating, t.vocal,
        t.intro_bars, t.outro_bars, t.variable_tempo, t.artist, t.title, t.location,
    )  # fmt: skip
    return hashlib.sha1(repr(fields).encode(), usedforsecurity=False).hexdigest()[:20]


def _canonical(value: Any) -> Any:
    """JSON-ready copy with sets sorted, so the result is identical in every process."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def config_key(cfg: ScoringConfig) -> str:
    """Identifies the scoring logic and weights a cached score was computed with."""
    canonical = json.dumps(_canonical(cfg.model_dump()), sort_keys=True)
    payload = f"{SCORER_VERSION}:{canonical}"
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:20]


class CompatibilityGraph:
    def __init__(
        self,
        tracks: Iterable[Track],
        cfg: ScoringConfig = DEFAULT_CONFIG,
        cache: dict[PairKey, PairScore] | None = None,
    ) -> None:
        self.cfg = cfg
        self.tracks = {t.id: t for t in tracks}
        self.fingerprints = {tid: track_fingerprint(t) for tid, t in self.tracks.items()}
        by_bpm = sorted((t.bpm, tid) for tid, t in self.tracks.items() if t.bpm)
        self._bpms = [bpm for bpm, _ in by_bpm]
        self._ids_by_bpm = [tid for _, tid in by_bpm]
        self._window = cfg.sets.tempo_window_pct / 100
        self._cache: dict[PairKey, PairScore] = dict(cache or {})
        self._candidates: dict[str, list[tuple[str, float]]] = {}
        self.new_scores: dict[PairKey, PairScore] = {}  # computed this run, for persisting

    def __len__(self) -> int:
        return len(self.tracks)

    def tempo_neighbors(self, a_id: str) -> list[str]:
        """Tracks b with |pitch change| <= the tempo window, trying a's BPM x1, x2 and /2."""
        a = self.tracks[a_id]
        if not a.bpm:
            return []
        found: set[str] = set()
        for multiplier in _TEMPO_MULTIPLIERS:
            target = a.bpm * multiplier
            # |target - b| / b <= w  <=>  target / (1 + w) <= b <= target / (1 - w)
            lo = bisect_left(self._bpms, target / (1 + self._window))
            hi = bisect_right(self._bpms, target / (1 - self._window))
            found.update(self._ids_by_bpm[lo:hi])
        found.discard(a_id)
        return sorted(found)

    def in_window(self, a_id: str, b_id: str) -> bool:
        a, b = self.tracks[a_id], self.tracks[b_id]
        if not a.bpm or not b.bpm:
            return False
        return any(abs(a.bpm * m - b.bpm) / b.bpm <= self._window for m in _TEMPO_MULTIPLIERS)

    def pair(self, a_id: str, b_id: str) -> PairScore:
        """(base points, energy boost) for a -> b, from memory, cache, or freshly scored."""
        key = (self.fingerprints[a_id], self.fingerprints[b_id])
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        score = score_transition(self.tracks[a_id], self.tracks[b_id], cfg=self.cfg)
        value = (
            score.total - score.components["energy"].points,
            Flag.ENERGY_BOOST_KEY in score.flags,
        )
        self._cache[key] = value
        self.new_scores[key] = value
        return value

    def candidates(self, a_id: str) -> list[tuple[str, float]]:
        """Tempo neighbors (b_id, upper bound on base), highest bound first.

        The bound never undercounts, so a caller may stop at the first neighbor whose bound
        cannot beat what it already has.
        """
        found = self._candidates.get(a_id)
        if found is None:
            a = self.tracks[a_id]
            found = [
                (b, base_upper_bound(a, self.tracks[b], self.cfg))
                for b in self.tempo_neighbors(a_id)
            ]
            found.sort(key=lambda e: (-e[1], e[0]))
            self._candidates[a_id] = found
        return found

"""Learn key-move and tempo-change habits from sets you actually played.

Sources, both yours:
- Rekordbox history (imported with `setsmith rekordbox import`): consecutive tracks in a
  history session are transitions you played.
- Live sets analyzed with `setsmith liveset analyze`.

The counts are blended into the scoring weights. Each learned value is how often you use
that move relative to your most-used one; it is mixed with the default with weight
n / (n + prior), so a handful of transitions barely moves anything and hundreds dominate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from setsmith.keys.camelot import KeyMove, classify_move, parse_key
from setsmith.model.track import Track
from setsmith.scoring.transition import tempo_match
from setsmith.scoring.weights import DEFAULT_CONFIG, Bands, ScoringConfig

BEYOND = "beyond"


def tempo_band_label(pct: float, bands: Bands) -> str:
    for max_value, _ in bands:
        if pct <= max_value:
            return f"<={max_value:g}%"
    return BEYOND


@dataclass(slots=True)
class LearnedPreferences:
    key_moves: dict[str, int] = field(default_factory=dict)
    tempo_bands: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)  # transitions per source
    prior: float = DEFAULT_CONFIG.liveset.learn_prior
    computed_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def key_count(self) -> int:
        return sum(self.key_moves.values())

    @property
    def tempo_count(self) -> int:
        return sum(self.tempo_bands.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPreferences:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def count(
    key_moves: Iterable[str | None],
    tempo_changes: Iterable[float | None],
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> tuple[Counter[str], Counter[str]]:
    keys = Counter(m for m in key_moves if m)
    tempos = Counter(tempo_band_label(t, cfg.tempo.bands) for t in tempo_changes if t is not None)
    return keys, tempos


def pair_features(a: Track, b: Track) -> tuple[str | None, float | None]:
    ka, kb = parse_key(a.camelot), parse_key(b.camelot)
    move = classify_move(ka, kb).value if ka and kb else None
    tempo = tempo_match(a.bpm, b.bpm)[1] if a.bpm and b.bpm else None
    return move, tempo


def history_pairs(
    sessions: Sequence[Sequence[str]], by_path: dict[str, Track]
) -> list[tuple[Track, Track]]:
    """Consecutive tracks of each history session (paths) that exist in the collection."""
    pairs = []
    for paths in sessions:
        tracks = [by_path.get(p) for p in paths]
        for a, b in pairwise(tracks):
            if a is not None and b is not None and a.id != b.id:
                pairs.append((a, b))
    return pairs


def learn(
    history: Sequence[tuple[Track, Track]],
    liveset_transitions: Sequence[tuple[str | None, float | None]],
    *,
    prior: float | None = None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> LearnedPreferences:
    features = [pair_features(a, b) for a, b in history] + list(liveset_transitions)
    keys, tempos = count((m for m, _ in features), (t for _, t in features), cfg)
    return LearnedPreferences(
        key_moves=dict(keys),
        tempo_bands=dict(tempos),
        sources={"history": len(history), "livesets": len(liveset_transitions)},
        prior=cfg.liveset.learn_prior if prior is None else prior,
    )


def _blend(default: float, learned: float, weight: float) -> float:
    return round((1 - weight) * default + weight * learned, 3)


def blended_config(prefs: LearnedPreferences, cfg: ScoringConfig = DEFAULT_CONFIG) -> ScoringConfig:
    """A copy of cfg with harmonic move scores and tempo band scores pulled toward prefs."""
    h, t = cfg.harmonic, cfg.tempo
    move_scores = dict(h.move_scores)
    n = prefs.key_count
    if n:
        weight = n / (n + prefs.prior)
        most = max(prefs.key_moves.values())
        move_scores = {
            move: _blend(score, prefs.key_moves.get(move.value, 0) / most, weight)
            for move, score in h.move_scores.items()
        }
    bands, beyond = t.bands, t.beyond_score
    n = prefs.tempo_count
    if n:
        weight = n / (n + prefs.prior)
        most = max(prefs.tempo_bands.values())
        bands = tuple(
            (limit, _blend(score, prefs.tempo_bands.get(f"<={limit:g}%", 0) / most, weight))
            for limit, score in t.bands
        )
        beyond = _blend(beyond, prefs.tempo_bands.get(BEYOND, 0) / most, weight)
    return cfg.model_copy(
        update={
            "harmonic": h.model_copy(update={"move_scores": move_scores}),
            "tempo": t.model_copy(update={"bands": bands, "beyond_score": beyond}),
        }
    )


def move_table(
    prefs: LearnedPreferences, cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[tuple[str, int, float, float]]:
    """(move, times played, default score, learned score) for display."""
    learned = blended_config(prefs, cfg).harmonic.move_scores
    return [
        (m.value, prefs.key_moves.get(m.value, 0), cfg.harmonic.move_scores[m], learned[m])
        for m in KeyMove
    ]


def tempo_table(
    prefs: LearnedPreferences, cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[tuple[str, int, float, float]]:
    learned = blended_config(prefs, cfg).tempo
    rows = [
        (f"<={limit:g}%", prefs.tempo_bands.get(f"<={limit:g}%", 0), default, new)
        for (limit, default), (_, new) in zip(cfg.tempo.bands, learned.bands, strict=True)
    ]
    rows.append(
        (BEYOND, prefs.tempo_bands.get(BEYOND, 0), cfg.tempo.beyond_score, learned.beyond_score)
    )
    return rows

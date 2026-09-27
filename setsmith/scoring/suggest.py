"""Rank the collection by how well each track follows a seed track."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.transition import TransitionScore, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

if TYPE_CHECKING:
    from setsmith.styles.profile import StyleFit, StyleProfile


@dataclass(frozen=True, slots=True)
class Suggestion:
    track: Track
    score: TransitionScore
    style_fit: StyleFit | None = None  # shown alongside, not added to the transition score


def suggest_next(
    collection: Collection,
    seed: Track,
    *,
    top: int = 10,
    energy_target: float | None = None,
    exclude_ids: Iterable[str] = (),
    cfg: ScoringConfig = DEFAULT_CONFIG,
    style: StyleProfile | None = None,
) -> list[Suggestion]:
    """Top `top` tracks to play after `seed`, best first.

    Skips the seed itself and other copies of the same song (same artist and title).
    """
    excluded = {seed.id, *exclude_ids}
    seed_song = collection.normalized_names(seed)
    results: list[Suggestion] = []
    for track in collection.tracks.values():
        if track.id in excluded:
            continue
        if track.title and collection.normalized_names(track) == seed_song:
            continue
        score = score_transition(seed, track, energy_target=energy_target, cfg=cfg, style=style)
        results.append(Suggestion(track, score))
    results.sort(key=lambda s: (-s.score.total, s.track.display, s.track.id))
    if style is None:
        return results[:top]
    return [Suggestion(s.track, s.score, style.style_fit(s.track, cfg)) for s in results[:top]]

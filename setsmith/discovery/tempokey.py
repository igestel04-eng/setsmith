"""BPM and key for tracks outside the collection: GetSongBPM first, then Deezer previews."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from setsmith.discovery.getsongbpm import GetSongBpm
from setsmith.discovery.http import ServiceError
from setsmith.discovery.preview import DeezerPreviews
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

GETSONGBPM = "GetSongBPM"
DEEZER_PREVIEW = "estimated from the Deezer preview"


@dataclass(frozen=True, slots=True)
class TempoKey:
    bpm: float | None
    camelot: str | None
    key_confidence: float
    source: str  # GETSONGBPM or DEEZER_PREVIEW
    links: dict[str, str]


def _safe_link(url: str) -> bool:
    return url.startswith(("https://", "http://"))


def lookup_tempo_key(
    songs: list[tuple[str, str]],
    bpm: GetSongBpm | None,
    previews: DeezerPreviews | None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[TempoKey | None], list[str]]:
    """BPM/key for each (artist, title), in order, and any warnings.

    A GetSongBPM error on the very first request (a bad key) is raised; later errors
    become warnings and the remaining songs go to Deezer.
    """
    results: list[TempoKey | None] = [None] * len(songs)
    notes: list[str] = []
    tag_confidence = cfg.harmonic.default_tag_confidence
    if bpm is not None:
        for i, (artist, title) in enumerate(songs):
            if on_progress:
                on_progress(f"looking up {artist} - {title} ({i + 1}/{len(songs)})")
            try:
                info = bpm.lookup(artist, title)
            except ServiceError as exc:
                if i == 0:
                    raise
                notes.append(str(exc))
                break  # stop hammering a failing service
            if info is not None:
                links = {"getsongbpm": info.url} if _safe_link(info.url) else {}
                results[i] = TempoKey(info.bpm, info.camelot, tag_confidence, GETSONGBPM, links)

    missing = [i for i, r in enumerate(results) if r is None or not (r.bpm and r.camelot)]
    if previews is None or not missing:
        return results, notes
    estimates, preview_notes = previews.lookup_many([songs[i] for i in missing], on_progress)
    notes += preview_notes
    key_confidence = cfg.discovery.preview_key_confidence
    for i, est in zip(missing, estimates, strict=True):
        if est is None or not est.found:
            continue
        known = results[i]
        links = dict(known.links) if known else {}
        if _safe_link(est.hit.link):
            links["deezer"] = est.hit.link
        if known is not None:  # GetSongBPM had one of the two: keep it, fill in the other
            results[i] = TempoKey(
                known.bpm or est.bpm,
                known.camelot or est.camelot,
                known.key_confidence if known.camelot else key_confidence,
                known.source if known.bpm and known.camelot else DEEZER_PREVIEW,
                links,
            )
        else:
            results[i] = TempoKey(est.bpm, est.camelot, key_confidence, DEEZER_PREVIEW, links)
    return results, notes

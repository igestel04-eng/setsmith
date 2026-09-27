"""Apply stored audio analysis to a loaded collection.

- intro_bars / outro_bars come from the analysis.
- Energy 1-10 is calibrated per library: each feature is turned into a z-score across the
  analyzed tracks, the weighted sum is ranked, and the rank is spread over 1-10. An energy
  tag in Comments ("E7", Mixed In Key's "Energy 7") still wins; audio replaces the
  star-rating fallback.
- The detected key is compared with the Rekordbox tag. Agreement keeps full confidence,
  adjacent or relative keys (common detector confusions) lower it a little, anything else
  lowers it a lot. Tracks without a usable tag take the detected key.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from setsmith.analysis.audio import TrackAnalysis, location_to_path
from setsmith.keys.camelot import KeyMove, classify_move, parse_key
from setsmith.model.collection import Collection
from setsmith.model.track import EnergySource, Track
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.store import Store

_RELATED_MOVES = {KeyMove.ADJACENT, KeyMove.RELATIVE}


@dataclass(slots=True)
class AnalysisCoverage:
    analyzed: int = 0  # current results applied
    stale: int = 0  # analyzed before, but the file changed since
    not_analyzed: int = 0
    missing_file: int = 0  # Location does not point to a readable local file
    key_agree: int = 0
    key_related: int = 0
    key_disagree: int = 0
    key_from_audio: int = 0  # no usable tag; detected key used
    energy_from_audio: int = 0
    structure_known: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def calibrate_energy(
    analyses: list[TrackAnalysis], cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[float]:
    """Energy 1-10 for each analysis, relative to the others in the list."""
    n = len(analyses)
    if n == 0:
        return []
    lo, hi = cfg.energy.scale_min, cfg.energy.scale_max
    if n == 1:
        return [round((lo + hi) / 2, cfg.analysis.energy_decimals)]

    composite = [0.0] * n
    for name, weight in cfg.analysis.energy_features.items():
        values = [a.feature(name) for a in analyses]
        known = [v for v in values if v is not None]
        if len(known) < 2:  # noqa: PLR2004 - a spread needs two values
            continue
        mean, spread = statistics.fmean(known), statistics.pstdev(known)
        for i, value in enumerate(values):
            if value is not None and spread > 0:
                composite[i] += weight * (value - mean) / spread

    # Average ranks for ties, spread over the scale.
    order = sorted(range(n), key=lambda i: composite[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and composite[order[j + 1]] == composite[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [round(lo + (hi - lo) * r / (n - 1), cfg.analysis.energy_decimals) for r in ranks]


def _apply_key(
    track: Track, analysis: TrackAnalysis, cfg: ScoringConfig, cov: AnalysisCoverage
) -> None:
    a = cfg.analysis
    detected = parse_key(analysis.key_camelot)
    track.detected_camelot = str(detected) if detected else None
    if detected is None:
        return
    tagged = parse_key(track.camelot)
    if tagged is None:
        track.camelot = str(detected)
        track.key_confidence = a.key_confidence_detected_only
        cov.key_from_audio += 1
        return
    move = classify_move(tagged, detected)
    if move == KeyMove.SAME:
        track.key_confidence = a.key_confidence_agree
        cov.key_agree += 1
    elif move in _RELATED_MOVES:
        track.key_confidence = a.key_confidence_related
        cov.key_related += 1
    else:
        track.key_confidence = a.key_confidence_disagree
        cov.key_disagree += 1


def apply_analyses(
    collection: Collection,
    analyses: dict[str, TrackAnalysis],
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> AnalysisCoverage:
    """Apply results (keyed by local path) to the collection's tracks, in place."""
    cov = AnalysisCoverage()
    matched: list[tuple[Track, TrackAnalysis]] = []
    for track in collection.tracks.values():
        path = location_to_path(track.location)
        analysis = analyses.get(str(path)) if path else None
        if analysis is not None:
            matched.append((track, analysis))

    energies = calibrate_energy([a for _, a in matched], cfg)
    for (track, analysis), energy in zip(matched, energies, strict=True):
        cov.analyzed += 1
        _apply_key(track, analysis, cfg, cov)
        if analysis.intro_bars is not None or analysis.outro_bars is not None:
            track.intro_bars, track.outro_bars = analysis.intro_bars, analysis.outro_bars
            cov.structure_known += 1
        if track.energy_source != EnergySource.COMMENT_TAG:
            track.energy, track.energy_source = energy, EnergySource.AUDIO
            cov.energy_from_audio += 1
    return cov


def local_files(collection: Collection) -> tuple[dict[str, tuple[str, float, int]], int]:
    """TrackID -> (path, mtime, size) for tracks whose file exists; and the missing count."""
    found: dict[str, tuple[str, float, int]] = {}
    missing = 0
    for track in collection.tracks.values():
        path = location_to_path(track.location)
        try:
            stat = path.stat() if path else None
        except OSError:
            stat = None
        if path is None or stat is None:
            missing += 1
            continue
        found[track.id] = (str(path), stat.st_mtime, stat.st_size)
    return found, missing


def load_and_apply(
    collection: Collection, store: Store, cfg: ScoringConfig = DEFAULT_CONFIG
) -> AnalysisCoverage:
    """Look up current analyses for the collection's files and apply them."""
    if store.count_analyses() == 0:
        return AnalysisCoverage(not_analyzed=len(collection))
    files, missing = local_files(collection)
    analyses = store.load_analyses(files.values(), cfg.analysis.version)
    cov = apply_analyses(collection, analyses, cfg)
    cov.missing_file = missing
    known = store.analyzed_paths()
    for path, _, _ in files.values():
        if path not in analyses:
            if path in known:
                cov.stale += 1
            else:
                cov.not_analyzed += 1
    return cov

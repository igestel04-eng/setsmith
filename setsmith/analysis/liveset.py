"""Analyze a recorded DJ set and/or its tracklist; draft a style profile from it.

Inputs are supplied by the user: a tracklist (text or CSV, see analysis/tracklist.py) and,
optionally, an audio recording of a set they are entitled to analyze. Nothing is fetched
or scraped. The recording is streamed from disk in blocks and never copied; only derived
numbers are kept.

Levels of detail, depending on what is available:

- Tracklist only: key moves, tempo changes, genre mix and energy order from the matched
  tracks' tags.
- + recording: windowed tempo, key and loudness every 30s; transition points from
  tracklist timestamps, or from novelty detection when there are none (approximate).
- + your original files for matched tracks: each original is aligned to the recording
  with beat-synchronous chroma + MFCC features and subsequence DTW (after Kim et al.,
  ISMIR 2020), giving where each track was cued in and out and how long each transition
  overlapped, in bars.
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from setsmith.analysis.audio import (
    AnalysisError,
    grid_beats,
    has_module,
    key_from_chroma,
    location_to_path,
    percentile,
)
from setsmith.analysis.tracklist import TracklistMatch, timestamps_usable
from setsmith.keys.camelot import KeyMove, classify_move, parse_key
from setsmith.model.track import Track
from setsmith.scoring.transition import tempo_match
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

_LENGTH_KEYS = ("8", "16", "32", "64")
_TYPE_KEYS = ("long_blend", "cut", "echo_out", "filter_sweep", "drop_swap")
_BEATS_PER_BAR = 4
_SECONDS_PER_MINUTE = 60.0
_EPS = 1e-9

# ---------------------------------------------------------------- records


@dataclass(slots=True)
class MatchRecord:
    """A tracklist line and the collection track it matched (if any)."""

    position: int
    raw: str
    artist: str
    title: str
    start_s: float | None
    unknown: bool
    together: bool
    track_id: str | None
    track_path: str | None  # local file path, so records survive re-exports
    track_display: str | None
    score: float
    # Snapshot of the matched track's tags, so stored sets need no collection later.
    bpm: float | None = None
    camelot: str | None = None
    genre: str = ""
    energy: float | None = None
    vocal: bool | None = None
    label: str = ""

    @classmethod
    def from_match(cls, m: TracklistMatch) -> MatchRecord:
        t = m.track
        path = location_to_path(t.location) if t else None
        return cls(
            position=m.entry.position,
            raw=m.entry.raw,
            artist=m.entry.artist,
            title=m.entry.title,
            start_s=m.entry.start_s,
            unknown=m.entry.unknown,
            together=m.entry.together,
            track_id=m.track.id if m.track else None,
            track_path=str(path) if path else None,
            track_display=t.display if t else None,
            score=round(m.score, 3),
            bpm=t.bpm if t else None,
            camelot=t.camelot if t else None,
            genre=t.genre if t else "",
            energy=t.energy if t else None,
            vocal=t.vocal if t else None,
            label=t.label if t else "",
        )


@dataclass(slots=True)
class Window:
    start_s: float
    bpm: float | None
    key: str | None
    rms_db: float


@dataclass(slots=True)
class Span:
    """Where a track sounds in the recording, and which part of the original that was."""

    position: int
    mix_start_s: float
    mix_end_s: float
    orig_start_s: float | None = None  # cue-in point in the original file
    orig_end_s: float | None = None  # cue-out point
    method: str = "timestamps"  # "dtw", "timestamps" or "novelty"
    cost: float | None = None  # mean alignment distance (dtw only)

    @property
    def approximate(self) -> bool:
        return self.method != "dtw"


@dataclass(slots=True)
class LiveTransition:
    from_position: int
    to_position: int
    from_id: str
    to_id: str
    key_move: str | None
    tempo_change_pct: float | None  # pitch change the incoming track needed (tags)
    overlap_s: float | None = None  # measured by alignment only
    overlap_bars: float | None = None
    cue_out_s: float | None = None  # in the outgoing original
    cue_in_s: float | None = None  # in the incoming original
    method: str = "tags"  # "dtw" when timings were measured

    @property
    def estimated_type(self) -> str | None:
        if self.overlap_bars is None:
            return None
        cfg = DEFAULT_CONFIG.liveset
        if self.overlap_bars >= cfg.blend_min_bars:
            return "long_blend"
        if self.overlap_bars >= cfg.sweep_min_bars:
            return "filter_sweep"
        return "cut"


@dataclass(slots=True)
class LiveSet:
    name: str
    tracklist_source: str
    audio_source: str | None
    duration_s: float | None
    matches: list[MatchRecord]
    windows: list[Window] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    transitions: list[LiveTransition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def matched(self) -> list[MatchRecord]:
        return [m for m in self.matches if m.track_id]

    @property
    def unmatched(self) -> list[MatchRecord]:
        return [m for m in self.matches if not m.track_id and not m.unknown]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveSet:
        def build(kind: Any, items: list[dict[str, Any]]) -> list[Any]:
            names = {f.name for f in fields(kind)}
            return [kind(**{k: v for k, v in item.items() if k in names}) for item in items]

        return cls(
            name=data["name"],
            tracklist_source=data["tracklist_source"],
            audio_source=data.get("audio_source"),
            duration_s=data.get("duration_s"),
            matches=build(MatchRecord, data.get("matches", [])),
            windows=build(Window, data.get("windows", [])),
            spans=build(Span, data.get("spans", [])),
            transitions=build(LiveTransition, data.get("transitions", [])),
            warnings=list(data.get("warnings", [])),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------- transitions and stats


def snap_bars(bars: float, cfg: ScoringConfig = DEFAULT_CONFIG) -> int | None:
    """Nearest standard transition length, or None for overlaps too short to count."""
    if bars < cfg.liveset.min_overlap_bars:
        return None
    allowed = cfg.transition.allowed_lengths_bars
    return min(allowed, key=lambda n: (abs(math.log2(n) - math.log2(bars)), n))


def build_transitions(
    matches: Sequence[MatchRecord],
    spans: Sequence[Span] = (),
    bpm_at: Callable[[float], float | None] | None = None,
) -> list[LiveTransition]:
    """Transitions between consecutive matched tracks (layered "w/" lines are skipped)."""
    by_position = {s.position: s for s in spans}
    playing = [m for m in matches if not m.together]
    out = []
    for a, b in pairwise(playing):
        if not a.track_id or not b.track_id:
            continue
        ka, kb = parse_key(a.camelot), parse_key(b.camelot)
        move = classify_move(ka, kb).value if ka and kb else None
        tempo = tempo_match(a.bpm, b.bpm)[1] if a.bpm and b.bpm else None
        tr = LiveTransition(a.position, b.position, a.track_id, b.track_id, move, tempo)
        sa, sb = by_position.get(a.position), by_position.get(b.position)
        if sa and sb and sa.method == "dtw" and sb.method == "dtw":
            overlap = max(0.0, sa.mix_end_s - sb.mix_start_s)
            bpm = (bpm_at(sb.mix_start_s) if bpm_at else None) or b.bpm or a.bpm
            tr.overlap_s = round(overlap, 2)
            if bpm:
                tr.overlap_bars = round(overlap * bpm / _SECONDS_PER_MINUTE / _BEATS_PER_BAR, 1)
            tr.cue_out_s, tr.cue_in_s = sa.orig_end_s, sb.orig_start_s
            tr.method = "dtw"
        out.append(tr)
    return out


def _track_bpms(ls: LiveSet) -> list[float]:
    """Tempo per matched track: measured in the recording when possible, else the tag."""
    spans = {s.position: s for s in ls.spans}
    bpms = []
    for m in ls.matched:
        measured = None
        span = spans.get(m.position)
        if span is not None and ls.windows:
            inside = [
                w.bpm
                for w in ls.windows
                if w.bpm and span.mix_start_s <= w.start_s < span.mix_end_s
            ]
            measured = statistics.median(inside) if inside else None
        value = measured or m.bpm
        if value:
            bpms.append(float(value))
    return bpms


def energy_trajectory(
    ls: LiveSet, cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[tuple[float, float]]:
    """(position 0-1, energy 1-10): recording loudness ranked within the set when there is
    a recording, else the matched tracks' energies in play order."""
    lo, hi = cfg.energy.scale_min, cfg.energy.scale_max
    if len(ls.windows) >= 2:  # noqa: PLR2004 - a trajectory needs two points
        levels = [w.rms_db for w in ls.windows]
        order = sorted(range(len(levels)), key=lambda i: levels[i])
        ranks = [0.0] * len(levels)
        for r, i in enumerate(order):
            ranks[i] = lo + (hi - lo) * r / (len(levels) - 1)
        k = cfg.liveset.energy_smooth_windows
        smooth = [
            statistics.fmean(ranks[max(0, i - k // 2) : i + k // 2 + 1]) for i in range(len(ranks))
        ]
        n = len(smooth)
        return [(i / (n - 1), round(v, 2)) for i, v in enumerate(smooth)]
    energies = [m.energy for m in ls.matched if m.energy is not None]
    if len(energies) < 2:  # noqa: PLR2004
        return []
    n = len(energies)
    return [(i / (n - 1), float(e)) for i, e in enumerate(energies)]


@dataclass(slots=True)
class LiveSetStats:
    matched: int
    unmatched: int
    unknown: int
    track_bpms: list[float]
    bpm_trajectory: list[tuple[float, float]]  # (seconds, bpm) from the recording
    key_moves: dict[str, int]
    tempo_changes_pct: list[float]
    energy: list[tuple[float, float]]
    transition_bars: dict[str, int]  # snapped measured overlaps
    transition_types: dict[str, int]  # estimated from overlaps
    genre_mix: dict[str, int]
    minor_share: float | None
    vocal_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_stats(ls: LiveSet, cfg: ScoringConfig = DEFAULT_CONFIG) -> LiveSetStats:
    matched = ls.matched
    keys = [k for m in matched if (k := parse_key(m.camelot))]
    vocals = [m.vocal for m in matched if m.vocal is not None]
    measured = [t for t in ls.transitions if t.overlap_bars is not None]
    bars = Counter(
        str(b) for t in measured if (b := snap_bars(t.overlap_bars or 0.0, cfg)) is not None
    )
    genres = Counter(g for m in matched for g in cfg.genre.normalize(m.genre))
    return LiveSetStats(
        matched=len(ls.matched),
        unmatched=len(ls.unmatched),
        unknown=sum(1 for m in ls.matches if m.unknown),
        track_bpms=_track_bpms(ls),
        bpm_trajectory=[(w.start_s, w.bpm) for w in ls.windows if w.bpm],
        key_moves=dict(Counter(t.key_move for t in ls.transitions if t.key_move)),
        tempo_changes_pct=[
            round(t.tempo_change_pct, 2) for t in ls.transitions if t.tempo_change_pct is not None
        ],
        energy=energy_trajectory(ls, cfg),
        transition_bars=dict(bars),
        transition_types=dict(Counter(kind for t in measured if (kind := t.estimated_type))),
        genre_mix=dict(genres.most_common()),
        minor_share=sum(k.is_minor for k in keys) / len(keys) if keys else None,
        vocal_share=sum(vocals) / len(vocals) if vocals else None,
    )


# ---------------------------------------------------------------- draft profile


def _shares(counts: dict[str, int], keys: Sequence[str]) -> dict[str, float] | None:
    total = sum(counts.get(k, 0) for k in keys)
    if total == 0:
        return None
    shares = {k: round(counts.get(k, 0) / total, 2) for k in keys}
    # Rounding can leave the sum at 0.99 or 1.01; put the remainder on the largest share.
    top = max(shares, key=lambda k: shares[k])
    shares[top] = round(shares[top] + 1 - sum(shares.values()), 2)
    return shares


def draft_profile(ls: LiveSet, cfg: ScoringConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """A style profile (StyleProfile JSON) drafted from a live set. Meant to be edited."""
    lc = cfg.liveset
    stats = compute_stats(ls, cfg)
    notes: list[str] = []

    bpms = stats.track_bpms or [120.0]
    p_lo, p_hi = lc.bpm_band_percentiles
    step = lc.bpm_round

    def rnd(v: float) -> float:
        return round(v / step) * step

    lo, hi = rnd(percentile(bpms, p_lo)), rnd(percentile(bpms, p_hi))
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    if not stats.track_bpms:
        notes.append("No tempo information; BPM band is a placeholder.")

    alpha = lc.move_smoothing
    move_names = [m.value for m in KeyMove if m != KeyMove.CLASH]
    most = max((stats.key_moves.get(m, 0) for m in move_names), default=0)
    allowed = {
        m: round((stats.key_moves.get(m, 0) + alpha) / (most + alpha), 2) for m in move_names
    }

    points = stats.energy
    curve: str | list[list[float]]
    if points:
        n = lc.curve_points
        samples: list[list[float]] = []
        for i in range(n):
            pos = i / (n - 1)
            nearest = min(points, key=lambda p: abs(p[0] - pos))
            samples.append([round(pos, 2), round(nearest[1], 1)])
        curve = samples
    else:
        curve = "journey"
        notes.append("No energy information; energy curve defaults to 'journey'.")

    genre_total = max(stats.genre_mix.values(), default=0)
    genre_weights = {g: round(c / genre_total, 2) for g, c in stats.genre_mix.items()}

    lengths = _shares(stats.transition_bars, _LENGTH_KEYS)
    types = _shares(stats.transition_types, _TYPE_KEYS)
    measured = sum(stats.transition_bars.values())
    if lengths is None or types is None:
        lengths, types = dict(lc.default_length_mix), dict(lc.default_type_mix)
        notes.append("Transition lengths and types are Setsmith defaults (nothing measured).")
    else:
        notes.append(
            f"Transition lengths from {measured} aligned transitions; types estimated from "
            "overlap length only (echo outs and drop swaps can't be told apart from cuts "
            "and sweeps)."
        )

    minor = stats.minor_share if stats.minor_share is not None else 0.5
    artists = Counter(n for m in ls.matched for n in [m.artist] if n)
    labels = Counter(m.label for m in ls.matched if m.label)
    notes.insert(
        0,
        f"Drafted by Setsmith from {stats.matched} matched of {len(ls.matches)} tracklist "
        f"entries and {len(ls.transitions)} transitions. Edit freely.",
    )
    sources = [f"tracklist: {Path(ls.tracklist_source).name}"]
    if ls.audio_source:
        sources.append(f"recording: {Path(ls.audio_source).name}")
    return {
        "name": f"Draft from {ls.name}",
        "description": f"Style parameters measured from the set '{ls.name}'.",
        "bpm_band": [lo, hi],
        "bpm_preferred": rnd(statistics.median(bpms)),
        "max_tempo_drift_bpm": max(1.0, round(max(bpms) - min(bpms), 1)),
        "key_mode_preference": {"minor": round(minor, 2), "major": round(1 - minor, 2)},
        "allowed_key_moves": allowed,
        "energy_curve": curve,
        "genre_weights": genre_weights,
        "vocal_density": round(stats.vocal_share, 2) if stats.vocal_share is not None else 0.5,
        "transition_length_bars": lengths,
        "transition_type_mix": types,
        "reference_artists": [a for a, _ in artists.most_common(lc.top_references)],
        "reference_labels": [lb for lb, _ in labels.most_common(lc.top_references)],
        "notes": " ".join(notes),
        "sources": sources,
    }


# ---------------------------------------------------------------- audio (needs numpy)


@dataclass(slots=True)
class MixFeatures:
    duration_s: float
    windows: list[Window]
    seg_times: Any  # start time (s) of each beat-synced column; column 0 is the pre-roll
    beat_features: Any  # (d x columns) unit-normalized chroma + MFCC, for DTW
    beat_mel: Any  # (n_mels x columns) mean mel power, for gain estimation


@dataclass(slots=True)
class TrackFeatures:
    seg_times: Any
    beat_features: Any
    beat_mel: Any


@dataclass(slots=True)
class DtwCore:
    """The part of a track DTW found playing 1:1, as mix column range plus a fixed offset
    (mix column = track column + offset)."""

    offset: int
    j0: int
    j1: int
    cost: float


def _require_audio() -> None:
    if not has_module("librosa"):
        raise AnalysisError("recording analysis needs the 'audio' extra: uv sync --extra audio")


def _frame_features(y: Any, cfg: ScoringConfig) -> tuple[Any, Any, Any]:
    """(chroma+MFCC frames, mel power frames, onset envelope) for one block of audio."""
    import librosa
    import numpy as np

    lc = cfg.liveset
    power = np.abs(librosa.stft(y, n_fft=lc.n_fft, hop_length=lc.hop_length)) ** 2
    chroma = librosa.feature.chroma_stft(S=power, sr=lc.sample_rate)
    mel = librosa.feature.melspectrogram(S=power, sr=lc.sample_rate, n_mels=lc.n_mels)
    mel_db = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=lc.n_mfcc)
    onset = librosa.onset.onset_strength(S=mel_db, sr=lc.sample_rate)
    return np.vstack([chroma, mfcc[1:]]), mel, onset


def _beat_sync(frames: Any, mel: Any, beat_frames: Any, cfg: ScoringConfig) -> tuple[Any, Any]:
    """Beat-synced (alignment features, mel power). Column k covers [beat k-1, beat k)."""
    import librosa
    import numpy as np

    synced = librosa.util.sync(frames, beat_frames, aggregate=np.median)
    chroma, mfcc = synced[:12], synced[12:]
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + _EPS)
    mfcc = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + _EPS)
    w = cfg.liveset.mfcc_weight
    feats = np.vstack([(1 - w) * chroma, w * mfcc])
    # Silent beats have no direction; give them a neutral one so cosine distance is defined.
    silent = np.linalg.norm(feats, axis=0) < _EPS
    feats[:, silent] = 1 / np.sqrt(feats.shape[0])
    return feats, librosa.util.sync(mel, beat_frames, aggregate=np.mean)


def _seg_times(beat_times: Any) -> Any:
    import numpy as np

    return np.concatenate([[0.0], beat_times])


def mix_features(
    path: Path, cfg: ScoringConfig = DEFAULT_CONFIG, use_essentia: bool = True
) -> MixFeatures:
    """Stream a recording in window-sized blocks: windowed tempo/key/RMS plus beat-synced
    alignment features. Memory stays bounded for multi-hour sets."""
    _require_audio()
    import librosa
    import numpy as np
    import soundfile as sf

    lc = cfg.liveset
    sr = lc.sample_rate
    essentia_key = None
    if use_essentia and has_module("essentia"):
        import essentia
        import essentia.standard as es

        essentia.log.warningActive = False
        essentia_key = es.KeyExtractor(profileType=cfg.analysis.essentia_key_profile, sampleRate=sr)

    windows: list[Window] = []
    frame_blocks, mel_blocks, onset_blocks = [], [], []
    try:
        handle = sf.SoundFile(str(path))
    except Exception as exc:  # soundfile raises its own error types
        raise AnalysisError(
            f"cannot stream {path.name}: {exc}. Convert the recording to WAV, FLAC or MP3."
        ) from exc
    with handle:
        native = handle.samplerate
        t = 0.0
        blocks = handle.blocks(blocksize=int(lc.window_s * native), always_2d=True, dtype="float32")
        for chunk in blocks:
            mono = chunk.mean(axis=1)
            y = librosa.resample(mono, orig_sr=native, target_sr=sr) if native != sr else mono
            if len(y) < sr:  # a sliver at the end
                t += len(y) / sr
                continue
            frames, mel, onset = _frame_features(y, cfg)
            frame_blocks.append(frames)
            mel_blocks.append(mel)
            onset_blocks.append(onset)
            if essentia_key is not None:
                k, scale, _ = essentia_key(np.asarray(y, dtype=np.float32))
                key = parse_key(f"{k} {scale}")
                key_text = str(key) if key else None
            else:
                key_text, _ = key_from_chroma(
                    [float(c) for c in frames[:12].mean(axis=1)], cfg.analysis
                )
            rms = float(10 * np.log10(np.mean(y.astype(np.float64) ** 2) + 1e-12))
            windows.append(Window(round(t, 2), None, key_text, round(rms, 2)))
            t += len(y) / sr

    if not frame_blocks:
        raise AnalysisError(f"{path.name} is too short to analyze")
    onset = np.concatenate(onset_blocks)
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=lc.hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=lc.hop_length)
    # Window tempo from the mean beat interval: frame-quantized tempo estimates on 30s
    # windows are several BPM coarse, but averaged beat intervals are not.
    for w in windows:
        inside = beat_times[(beat_times >= w.start_s) & (beat_times < w.start_s + lc.window_s)]
        if len(inside) > 2:  # noqa: PLR2004 - an interval average needs a few beats
            w.bpm = round(float(60.0 / np.mean(np.diff(inside))), 2)
    feats, mel = _beat_sync(np.hstack(frame_blocks), np.hstack(mel_blocks), beats, cfg)
    return MixFeatures(t, windows, _seg_times(beat_times), feats, mel)


def track_features(path: Path, track: Track, cfg: ScoringConfig = DEFAULT_CONFIG) -> TrackFeatures:
    """Beat-synced features for an original file, on its Rekordbox grid when it has one."""
    _require_audio()
    import librosa
    import numpy as np

    lc = cfg.liveset
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=lc.sample_rate, mono=True)
    except Exception as exc:
        raise AnalysisError(f"could not decode {path.name}: {exc}") from exc
    frames, mel, onset = _frame_features(y, cfg)
    duration = len(y) / lc.sample_rate
    if track.beat_grid:
        times = np.array([t for t, _ in grid_beats(track.beat_grid, duration)])
        beats = librosa.time_to_frames(times, sr=lc.sample_rate, hop_length=lc.hop_length)
    else:
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=lc.sample_rate,
            hop_length=lc.hop_length,
            start_bpm=track.bpm or 120.0,
        )
        times = librosa.frames_to_time(beats, sr=lc.sample_rate, hop_length=lc.hop_length)
    keep = beats < frames.shape[1]
    beats, times = beats[keep], times[keep]
    feats, beat_mel = _beat_sync(frames, mel, beats, cfg)
    return TrackFeatures(_seg_times(times), feats, beat_mel)


def played_region(
    path_pairs: Sequence[tuple[int, int]], cfg: ScoringConfig = DEFAULT_CONFIG
) -> tuple[int, int] | None:
    """Index range of the longest stretch of a DTW path that advances 1:1 (the part of the
    track that was actually playing at normal speed). None if too short."""
    lc = cfg.liveset
    steps = [(b[0] - a[0], b[1] - a[1]) == (1, 1) for a, b in pairwise(path_pairs)]
    if not steps:
        return None
    w = lc.diagonal_window_beats
    good = []
    for i in range(len(steps)):
        window = steps[max(0, i - w // 2) : i + w // 2 + 1]
        good.append(sum(window) / len(window) >= lc.diagonal_min_share)
    best: tuple[int, int] | None = None
    start = None
    for i, ok in enumerate([*good, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None or best[1] - best[0] < lc.min_played_beats:
        return None
    return best


def dtw_core(
    track: TrackFeatures,
    mix: MixFeatures,
    search: tuple[float, float],
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> DtwCore | None:
    """Find where a track plays at normal speed inside the recording (subsequence DTW)."""
    import librosa
    import numpy as np

    lc = cfg.liveset
    lo, hi = (int(v) for v in np.searchsorted(mix.seg_times, list(search)))
    region = mix.beat_features[:, lo:hi]
    if region.shape[1] < lc.min_played_beats or track.beat_features.shape[1] < lc.min_played_beats:
        return None
    w = lc.dtw_offdiagonal_weight
    _, path = librosa.sequence.dtw(
        X=track.beat_features,
        Y=region,
        metric="cosine",
        subseq=True,
        step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
        weights_mul=np.array([1.0, w, w]),
    )
    pairs = [(int(i), int(j)) for i, j in path[::-1]]
    found = played_region(pairs, cfg)
    if found is None:
        return None
    stretch = pairs[found[0] : found[1] + 1]
    a = track.beat_features[:, [i for i, _ in stretch]]
    b = region[:, [j for _, j in stretch]]
    cost = float(
        np.mean(
            1 - (a * b).sum(axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + _EPS)
        )
    )
    if cost > lc.max_match_cost:
        return None
    offset = int(np.median([j + lo - i for i, j in stretch]))
    return DtwCore(offset, stretch[0][1] + lo, stretch[-1][1] + lo, round(cost, 3))


def estimate_gains(
    items: Sequence[tuple[TrackFeatures, DtwCore]],
    mix: MixFeatures,
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> list[Any]:
    """Per-beat power gain of each track in the mix: mix mel ~ sum of gain x track mel,
    solved by non-negative least squares over the tracks that could be playing."""
    import numpy as np
    from scipy.optimize import nnls

    lc = cfg.liveset
    n_mix = mix.beat_mel.shape[1]
    # Level-normalize each track so a gain of ~1 means "as loud as in its DTW core".
    scaled = []
    for track, core in items:
        mel = track.beat_mel / (np.median(np.linalg.norm(track.beat_mel, axis=0)) + _EPS)
        scaled.append((mel, core))
    gains = [np.full(n_mix, np.nan) for _ in items]
    for j in range(n_mix):
        active = [
            (k, j - core.offset)
            for k, (mel, core) in enumerate(scaled)
            if 0 <= j - core.offset < mel.shape[1]
            and core.j0 - lc.max_extend_beats <= j <= core.j1 + lc.max_extend_beats
        ]
        if not active:
            continue
        basis = np.column_stack([scaled[k][0][:, i] for k, i in active])
        solution, _ = nnls(basis, mix.beat_mel[:, j])
        for (k, _), g in zip(active, solution, strict=True):
            gains[k][j] = g
    return gains


def _fade_edge(amp: Any, start: int, step: int, cfg: ScoringConfig) -> int:
    """Index where a fade out of the core (direction `step`) reaches silence.

    Walks outwards while the relative amplitude stays above the floor and keeps falling,
    then extrapolates a straight line fitted to the middle of the fade down to zero.
    """
    import numpy as np

    lc = cfg.liveset
    lo, hi = lc.fade_fit_range
    j, lowest = start, float("inf")
    points: list[tuple[int, float]] = []
    while 0 <= j + step < len(amp) and abs(j + step - start) <= lc.max_extend_beats:
        value = amp[j + step]
        if np.isnan(value) or value < lc.fade_floor or value > lowest + lc.fade_rise_tolerance:
            break
        j += step
        lowest = min(lowest, float(value))
        if lo <= value <= hi:
            points.append((j, float(value)))
    if len(points) >= lc.fade_min_points:
        x, y = np.array(points).T
        slope, intercept = np.polyfit(x, y, 1)
        if slope * step < 0:  # amplitude falls moving away from the core
            zero = round(float(-intercept / slope))
            limit = start + step * lc.max_extend_beats
            return max(zero, limit) if step < 0 else min(zero, limit)
    return j


def refine_span(
    position: int,
    track: TrackFeatures,
    core: DtwCore,
    gains: Any,
    mix: MixFeatures,
    cfg: ScoringConfig = DEFAULT_CONFIG,
) -> Span:
    """Grow a DTW core over the fades on either side, using the track's estimated gains."""
    import numpy as np

    k = cfg.liveset.amp_smooth_beats
    amp = np.sqrt(np.clip(gains, 0, None))
    smooth = np.full(len(amp), np.nan)
    for j in range(len(amp)):
        window = amp[max(0, j - k // 2) : j + k // 2 + 1]
        if not np.all(np.isnan(window)):
            smooth[j] = np.nanmedian(window)
    core_values = smooth[core.j0 : core.j1 + 1]
    core_values = core_values[~np.isnan(core_values)]
    reference = float(np.median(core_values)) if len(core_values) else 0.0
    # Without usable gains the edges stay at the DTW core.
    smooth = np.full(len(amp), np.nan) if reference <= _EPS else smooth / reference

    last = len(mix.seg_times) - 1
    n_track = len(track.seg_times)
    # A track cannot sound before its first beat or after its last one.
    first, final = max(0, core.offset), min(last, core.offset + n_track - 1)
    j0 = max(first, _fade_edge(smooth, core.j0, -1, cfg))
    j1 = min(final, _fade_edge(smooth, core.j1, +1, cfg))

    def orig_time(j: int) -> float:
        return float(track.seg_times[min(max(j - core.offset, 0), n_track - 1)])

    end_s = float(mix.seg_times[j1 + 1]) if j1 + 1 <= last else mix.duration_s
    return Span(
        position=position,
        mix_start_s=round(float(mix.seg_times[j0]), 2),
        mix_end_s=round(end_s, 2),
        orig_start_s=round(orig_time(j0), 2),
        orig_end_s=round(orig_time(j1 + 1), 2),
        method="dtw",
        cost=core.cost,
    )


def novelty_boundaries(
    mix: MixFeatures, count: int, cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[float]:
    """`count` change points in the recording (bar-level feature novelty), in seconds."""
    import numpy as np

    feats = mix.beat_features
    n_bars = feats.shape[1] // _BEATS_PER_BAR
    if count <= 0 or n_bars < 2 * cfg.liveset.novelty_half_window_bars:
        return []
    bars = (
        feats[:, : n_bars * _BEATS_PER_BAR]
        .reshape(feats.shape[0], n_bars, _BEATS_PER_BAR)
        .mean(axis=2)
    )
    k = cfg.liveset.novelty_half_window_bars
    novelty = np.zeros(n_bars)
    for t in range(k, n_bars - k):
        before, after = bars[:, t - k : t].mean(axis=1), bars[:, t : t + k].mean(axis=1)
        novelty[t] = 1 - np.dot(before, after) / (
            np.linalg.norm(before) * np.linalg.norm(after) + _EPS
        )
    min_gap = max(1, n_bars // (count + 1) // 2)
    chosen: list[int] = []
    for t in np.argsort(novelty)[::-1]:
        if novelty[t] <= 0:
            break
        if all(abs(int(t) - c) >= min_gap for c in chosen):
            chosen.append(int(t))
        if len(chosen) == count:
            break
    bar_times = mix.seg_times[1::_BEATS_PER_BAR]
    return sorted(float(bar_times[c]) for c in chosen if c < len(bar_times))


# ---------------------------------------------------------------- pipeline


def analyze_liveset(
    name: str,
    tracklist_source: str,
    matches: list[TracklistMatch],
    *,
    audio: Path | None = None,
    align: bool = True,
    use_essentia: bool = True,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> LiveSet:
    """Everything the inputs allow; see the module docstring for the levels of detail."""
    records = [MatchRecord.from_match(m) for m in matches]
    ls = LiveSet(name, tracklist_source, str(audio) if audio else None, None, records)
    if ls.unmatched:
        ls.warnings.append(f"{len(ls.unmatched)} tracklist line(s) not found in the collection")

    mix: MixFeatures | None = None
    if audio is not None:
        if on_progress:
            on_progress("reading the recording")
        mix = mix_features(audio, cfg, use_essentia)
        ls.duration_s = round(mix.duration_s, 1)
        ls.windows = mix.windows

    if mix is not None:
        ls.spans = _find_spans(ls, matches, mix, align, cfg, on_progress)

    def bpm_at(t: float) -> float | None:
        near = [w.bpm for w in ls.windows if w.bpm and abs(w.start_s - t) <= cfg.liveset.window_s]
        return statistics.median(near) if near else None

    ls.transitions = build_transitions(records, ls.spans, bpm_at if ls.windows else None)
    return ls


def _find_spans(
    ls: LiveSet,
    matches: list[TracklistMatch],
    mix: MixFeatures,
    align: bool,
    cfg: ScoringConfig,
    on_progress: Callable[[str], None] | None,
) -> list[Span]:
    lc = cfg.liveset
    playing = [m for m in matches if not m.entry.together]
    use_times = timestamps_usable([m.entry for m in playing])

    if align:
        floor = 0.0
        found: list[tuple[int, TrackFeatures, DtwCore]] = []
        for idx, m in enumerate(playing):
            if m.track is None:
                continue
            path = location_to_path(m.track.location)
            if path is None or not path.is_file():
                continue
            if use_times:
                start = m.entry.start_s or 0.0
                nxt = playing[idx + 1].entry.start_s if idx + 1 < len(playing) else mix.duration_s
                search = (
                    max(0.0, start - lc.search_margin_s),
                    min(mix.duration_s, (nxt or start) + lc.search_margin_s),
                )
            else:
                search = (floor, min(mix.duration_s, floor + lc.max_search_s))
            if on_progress:
                on_progress(f"aligning {m.track.display}")
            try:
                features = track_features(path, m.track, cfg)
            except AnalysisError as exc:
                ls.warnings.append(str(exc))
                continue
            core = dtw_core(features, mix, search, cfg)
            if core is None:
                ls.warnings.append(f"could not locate {m.track.display} in the recording")
                continue
            found.append((m.entry.position, features, core))
            floor = float(mix.seg_times[core.j0])
        if found:
            if on_progress:
                on_progress("estimating fades")
            gains = estimate_gains([(f, c) for _, f, c in found], mix, cfg)
            return [
                refine_span(pos, f, c, g, mix, cfg)
                for (pos, f, c), g in zip(found, gains, strict=True)
            ]
        ls.warnings.append("no track could be aligned; transition points are approximate")

    if use_times:
        times_list = [m.entry.start_s or 0.0 for m in playing]
    else:
        cuts = novelty_boundaries(mix, len(playing) - 1, cfg)
        times_list = [0.0, *cuts]
        if len(times_list) != len(playing):
            ls.warnings.append("could not find a boundary for every track")
    ends = [*times_list[1:], mix.duration_s]
    method = "timestamps" if use_times else "novelty"
    return [
        Span(m.entry.position, round(s, 2), round(e, 2), method=method)
        for m, s, e in zip(playing, times_list, ends, strict=False)
    ]

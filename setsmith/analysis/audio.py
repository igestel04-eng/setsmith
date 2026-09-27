"""Local audio analysis: key, loudness, energy features and intro/outro length.

Backends, all optional (install the `audio` extra, plus `essentia` for the best keys):

- librosa (required for analysis): decoding fallback, RMS loudness, onset rate, spectral
  centroid and flux, beat tracking, per-bar energy for intro/outro, fallback key detection
  (Krumhansl-Kessler template matching on chroma).
- Essentia: key with EDM profiles ("edma" by default), EBU R128 loudness, danceability.
- allin1 (installed separately): musical structure segments for intro/outro.

Only local files are read, never modified. Results are derived numbers only.
Pure helpers in this module avoid numpy so they work without the audio extras.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from setsmith.keys.camelot import CAMELOT_WHEEL_SIZE, parse_key
from setsmith.model.track import TempoMarker
from setsmith.scoring.weights import DEFAULT_CONFIG, AnalysisConfig

Structure = Literal["grid", "allin1"]
_NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_ALLIN1_SILENCE = {"start", "end"}
_WINDOWS_DRIVE_LEN = 3  # "/C:"


class AnalysisError(RuntimeError):
    """A file could not be analyzed (unreadable, too short, backend missing)."""


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def location_to_path(location: str) -> Path | None:
    """Local path for a Rekordbox Location URI ("file://localhost/Users/..."), else None."""
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.netloc not in ("", "localhost"):
        return None
    path = unquote(parsed.path if parsed.scheme else location)
    # Windows exports look like file://localhost/C:/Music/...
    if len(path) >= _WINDOWS_DRIVE_LEN and path[0] == "/" and path[2] == ":" and path[1].isalpha():
        path = path[1:]
    return Path(path)


# ---------------------------------------------------------------- result


@dataclass(slots=True)
class TrackAnalysis:
    path: str
    mtime: float
    size: int
    version: int
    analyzed_at: str
    duration_s: float
    loudness_db: float  # integrated RMS, dBFS
    onset_rate: float  # onsets per second
    spectral_centroid_hz: float
    spectral_flux: float  # mean onset strength
    key_camelot: str | None = None
    key_strength: float | None = None
    key_source: str | None = None  # "essentia:edma", "librosa:kk"
    loudness_lufs: float | None = None  # EBU R128 integrated, Essentia with --lufs only
    danceability: float | None = None  # Essentia only
    bpm_detected: float | None = None
    bars: int | None = None
    intro_bars: int | None = None
    outro_bars: int | None = None
    structure_source: str | None = None  # "rekordbox_grid", "librosa_beats", "allin1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackAnalysis:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def feature(self, name: str) -> float | None:
        value = getattr(self, name, None)
        return float(value) if isinstance(value, int | float) else None


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    track_id: str
    path: str
    markers: tuple[TempoMarker, ...] = ()
    tag_bpm: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    essentia: bool = True  # use Essentia when installed
    key_profile: str = DEFAULT_CONFIG.analysis.essentia_key_profile
    structure: Structure = "grid"
    lufs: bool = False  # EBU R128 integrated loudness (Essentia; ~3s per track, not scored)
    cfg: AnalysisConfig = DEFAULT_CONFIG.analysis


DEFAULT_OPTIONS = AnalysisOptions()


# ---------------------------------------------------------------- pure helpers


def _beats_per_bar(marker: TempoMarker, default: int) -> int:
    try:
        return max(1, int(marker.metro.split("/")[0]))
    except (ValueError, IndexError):
        return default


def grid_beats(
    markers: Sequence[TempoMarker],
    duration_s: float,
    default_beats_per_bar: int = DEFAULT_CONFIG.analysis.default_beats_per_bar,
) -> list[tuple[float, int]]:
    """(time, beat number in bar) for every beat of a Rekordbox grid, from 0s to the end."""
    ordered = sorted((m for m in markers if m.bpm > 0), key=lambda m: m.inizio_s)
    if not ordered:
        return []
    beats: list[tuple[float, int]] = []

    first = ordered[0]
    bpb = _beats_per_bar(first, default_beats_per_bar)
    spb = 60.0 / first.bpm
    beat, k = first.battito, 1
    while first.inizio_s - k * spb >= 0:
        beat = (beat - 2) % bpb + 1  # the beat number one beat earlier
        beats.append((first.inizio_s - k * spb, beat))
        k += 1
    beats.reverse()

    for i, marker in enumerate(ordered):
        end = ordered[i + 1].inizio_s if i + 1 < len(ordered) else duration_s
        bpb = _beats_per_bar(marker, default_beats_per_bar)
        spb = 60.0 / marker.bpm
        beat, k = marker.battito, 0
        while (t := marker.inizio_s + k * spb) < end - 1e-6:
            beats.append((t, beat))
            beat = beat % bpb + 1
            k += 1
    return beats


def bar_starts(
    markers: Sequence[TempoMarker],
    duration_s: float,
    default_beats_per_bar: int = DEFAULT_CONFIG.analysis.default_beats_per_bar,
) -> list[float]:
    """Downbeat times from a Rekordbox beat grid, including bars before the first marker."""
    return [t for t, beat in grid_beats(markers, duration_s, default_beats_per_bar) if beat == 1]


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    pos = (len(ordered) - 1) * pct / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _snap(bars: int, phrase: int) -> int:
    return phrase * round(bars / phrase)


def detect_sections(
    levels_db: Sequence[float], cfg: AnalysisConfig = DEFAULT_CONFIG.analysis
) -> tuple[int | None, int | None]:
    """(intro_bars, outro_bars) from per-bar levels: bars before the first sustained full
    section and after the last one, snapped to phrases. None where detection is implausible.
    """
    n, k = len(levels_db), cfg.full_min_bars
    if n < 2 * k:
        return None, None
    threshold = percentile(levels_db, cfg.full_level_percentile) - cfg.full_margin_db
    full = [level >= threshold for level in levels_db]
    first = next((i for i in range(n - k + 1) if all(full[i : i + k])), None)
    last = next((j for j in range(n - 1, k - 2, -1) if all(full[j - k + 1 : j + 1])), None)
    if first is None or last is None:
        return None, None
    limit = n * cfg.max_section_fraction
    intro = _snap(first, cfg.phrase_bars)
    outro = _snap(n - 1 - last, cfg.phrase_bars)
    return (intro if intro <= limit else None), (outro if outro <= limit else None)


def structure_from_segments(
    downbeats: Sequence[float], segments: Sequence[tuple[float, float, str]]
) -> tuple[int | None, int | None]:
    """(intro_bars, outro_bars) from labeled segments such as allin1's."""
    body = [s for s in segments if s[2].lower() not in _ALLIN1_SILENCE]
    if not body or not downbeats:
        return None, None
    main_start = next((s[0] for s in body if s[2].lower() != "intro"), body[-1][1])
    intro = sum(1 for d in downbeats if d < main_start)
    outro = 0
    if body[-1][2].lower() == "outro":
        outro_start = body[-1][0]
        for seg in reversed(body):
            if seg[2].lower() != "outro":
                break
            outro_start = seg[0]
        outro = sum(1 for d in downbeats if d >= outro_start)
    return intro, outro


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (vx * vy) if vx and vy else 0.0


def key_from_chroma(
    chroma: Sequence[float], cfg: AnalysisConfig = DEFAULT_CONFIG.analysis
) -> tuple[str | None, float]:
    """(Camelot key, correlation) by Krumhansl-Kessler template matching. chroma[0] is C."""
    if len(chroma) != CAMELOT_WHEEL_SIZE or not any(chroma):
        return None, 0.0
    best: tuple[float, int, str] = (-2.0, 0, "major")
    for tonic in range(CAMELOT_WHEEL_SIZE):
        for mode, profile in (("major", cfg.kk_major), ("minor", cfg.kk_minor)):
            rotated = [profile[(pc - tonic) % CAMELOT_WHEEL_SIZE] for pc in range(12)]
            best = max(best, (_pearson(chroma, rotated), tonic, mode))
    corr, tonic, mode = best
    key = parse_key(f"{_NOTE_NAMES[tonic]} {mode}")
    return (str(key) if key else None), corr


# ---------------------------------------------------------------- analysis (needs numpy)


def analyze_file(job: AnalysisJob, options: AnalysisOptions = DEFAULT_OPTIONS) -> TrackAnalysis:
    """Analyze one local file. Raises AnalysisError when it cannot."""
    cfg = options.cfg
    path = Path(job.path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if not has_module("librosa"):
        raise AnalysisError("audio analysis needs the 'audio' extra: uv sync --extra audio")

    import librosa
    import numpy as np

    use_essentia = options.essentia and has_module("essentia")
    key_camelot: str | None = None
    key_strength: float | None = None
    key_source: str | None = None
    lufs: float | None = None
    danceability: float | None = None
    sr = cfg.sample_rate

    try:
        if use_essentia:
            import essentia
            import essentia.standard as es

            essentia.log.warningActive = False
            sr_hi = cfg.essentia_sample_rate
            y_hi = es.MonoLoader(filename=str(path), sampleRate=sr_hi)()
            if len(y_hi) / sr_hi < cfg.min_duration_s:
                raise AnalysisError(f"shorter than {cfg.min_duration_s:g}s")
            key, scale, strength = es.KeyExtractor(
                profileType=options.key_profile, sampleRate=sr_hi
            )(y_hi)
            parsed = parse_key(f"{key} {scale}")
            key_camelot = str(parsed) if parsed else None
            key_strength, key_source = float(strength), f"essentia:{options.key_profile}"
            if options.lufs:
                stereo, native_sr, *_ = es.AudioLoader(filename=str(path))()
                lufs = float(es.LoudnessEBUR128(sampleRate=native_sr)(stereo)[2])
            danceability = float(es.Danceability(sampleRate=sr_hi)(y_hi)[0])
            y = librosa.resample(np.asarray(y_hi, dtype=np.float32), orig_sr=sr_hi, target_sr=sr)
        else:
            with warnings.catch_warnings():  # librosa warns before its audioread fallback
                warnings.simplefilter("ignore")
                y, _ = librosa.load(str(path), sr=sr, mono=True)
    except AnalysisError:
        raise
    except Exception as exc:  # decoders raise many types; report them uniformly
        raise AnalysisError(f"could not decode {path.name}: {exc}") from exc

    duration = len(y) / sr
    if duration < cfg.min_duration_s:
        raise AnalysisError(f"shorter than {cfg.min_duration_s:g}s")

    hop = cfg.hop_length
    spectrum = np.abs(librosa.stft(y, hop_length=hop))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, start_bpm=job.tag_bpm or 120.0
    )
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sr)

    if key_camelot is None:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).mean(axis=1)
        key_camelot, corr = key_from_chroma([float(c) for c in chroma], cfg)
        key_strength, key_source = corr, "librosa:kk"

    intro = outro = n_bars = None
    structure_source: str | None = None
    if options.structure == "allin1":
        intro, outro, n_bars = _allin1_structure(path)
        structure_source = "allin1"
    else:
        if job.markers:
            starts = bar_starts(job.markers, duration, cfg.default_beats_per_bar)
            structure_source = "rekordbox_grid"
        else:
            # No grid: tempo from the tag (beat trackers often halve sparse tracks), phase
            # from the first tracked beat.
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
            bpm = job.tag_bpm or float(np.atleast_1d(tempo)[0])
            first = float(beat_times[0]) if len(beat_times) else 0.0
            marker = TempoMarker(first, bpm, f"{cfg.default_beats_per_bar}/4", 1)
            starts = bar_starts([marker], duration, cfg.default_beats_per_bar)
            structure_source = "librosa_beats"
        levels = _bar_levels(y, starts, duration, cfg)
        n_bars = len(levels)
        intro, outro = detect_sections(levels, cfg)

    return TrackAnalysis(
        path=str(path),
        mtime=stat.st_mtime,
        size=stat.st_size,
        version=cfg.version,
        analyzed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=round(duration, 3),
        loudness_db=float(10 * np.log10(np.mean(y**2) + 1e-12)),
        onset_rate=len(onsets) / duration,
        spectral_centroid_hz=float(centroid.mean()),
        spectral_flux=float(onset_env.mean()),
        key_camelot=key_camelot,
        key_strength=key_strength,
        key_source=key_source,
        loudness_lufs=lufs,
        danceability=danceability,
        bpm_detected=float(np.atleast_1d(tempo)[0]),
        bars=n_bars,
        intro_bars=intro,
        outro_bars=outro,
        structure_source=structure_source,
    )


def _bar_levels(y: Any, starts: list[float], duration: float, cfg: AnalysisConfig) -> list[float]:
    """Harmonic energy (dB) per bar."""
    import librosa
    import numpy as np

    sr, hop = cfg.sample_rate, cfg.structure_hop
    spectrum = np.abs(librosa.stft(y, n_fft=cfg.structure_n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.structure_n_fft)
    harmonic, _ = librosa.decompose.hpss(
        spectrum[freqs <= cfg.structure_max_hz], kernel_size=cfg.hpss_kernel
    )
    power = (harmonic**2).sum(axis=0)
    frame_times = librosa.frames_to_time(np.arange(len(power)), sr=sr, hop_length=hop)
    ends = [*starts[1:], duration]
    lengths = [e - s for s, e in zip(starts, ends, strict=True)]
    typical = sorted(lengths)[len(lengths) // 2] if lengths else 0.0
    levels = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < typical * cfg.min_bar_fraction:
            continue
        a, b = np.searchsorted(frame_times, [start, end])
        mean = float(power[a:b].mean()) if b > a else 0.0
        levels.append(10 * math.log10(mean) if mean > 0 else cfg.level_floor_db)
    return levels


def _allin1_structure(path: Path) -> tuple[int | None, int | None, int | None]:
    if not has_module("allin1"):
        raise AnalysisError("--structure allin1 needs allin1 installed (see README)")
    import allin1

    result = allin1.analyze(str(path))
    segments = [(float(s.start), float(s.end), str(s.label)) for s in result.segments]
    downbeats = [float(d) for d in result.downbeats]
    intro, outro = structure_from_segments(downbeats, segments)
    return intro, outro, len(downbeats)


# ---------------------------------------------------------------- batch


def _run_job(job: AnalysisJob, options: AnalysisOptions) -> TrackAnalysis:
    return analyze_file(job, options)


def analyze_many(
    jobs: Iterable[AnalysisJob],
    options: AnalysisOptions = DEFAULT_OPTIONS,
    *,
    workers: int = 1,
    on_start: Callable[[AnalysisJob], None] | None = None,
) -> Iterator[tuple[AnalysisJob, TrackAnalysis | AnalysisError]]:
    """Analyze jobs, yielding each result (or its error) as soon as it is ready.

    workers > 1 uses separate processes; each loads the audio libraries once.
    """
    job_list = list(jobs)
    if workers <= 1:
        for job in job_list:
            if on_start:
                on_start(job)
            try:
                yield job, analyze_file(job, options)
            except AnalysisError as exc:
                yield job, exc
        return

    context = "spawn" if sys.platform == "darwin" else None
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context(context)
    ) as pool:
        pending: dict[Future[TrackAnalysis], AnalysisJob] = {}
        queue = iter(job_list)

        def submit_next() -> None:
            job = next(queue, None)
            if job is not None:
                if on_start:
                    on_start(job)
                pending[pool.submit(_run_job, job, options)] = job

        for _ in range(workers * 2):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    yield job, future.result()
                except AnalysisError as exc:
                    yield job, exc
                except Exception as exc:  # a crashed worker must not stop the batch
                    yield job, AnalysisError(f"{type(exc).__name__}: {exc}")
                submit_next()


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)

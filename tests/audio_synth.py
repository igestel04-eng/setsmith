"""Synthesized house tracks with known key, tempo and structure, for analysis tests.

Needs numpy and soundfile (the `audio` extra). Everything is generated: no real recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

FloatArray = npt.NDArray[np.float64]

# Chords as MIDI notes, two bars each. A harmonic minor: Am - Dm - E - Am.
A_MINOR_PROGRESSION = [(57, 60, 64), (62, 65, 69), (64, 68, 71), (57, 60, 64)]
A_MINOR_BASS = [45, 50, 52, 45]  # A2, D3, E3, A2


def _hz(midi: int) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12))


def _add(out: FloatArray, start_s: float, sound: FloatArray, sr: int) -> None:
    i = int(start_s * sr)
    j = min(len(out), i + len(sound))
    if i < len(out):
        out[i:j] += sound[: j - i]


def _kick(sr: int) -> FloatArray:
    t = np.arange(int(0.25 * sr)) / sr
    freq = 50 + 70 * np.exp(-t * 30)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.asarray(0.9 * np.sin(phase) * np.exp(-t * 12), dtype=np.float64)


def _hat(sr: int, rng: np.random.Generator) -> FloatArray:
    n = int(0.04 * sr)
    noise = np.diff(rng.standard_normal(n + 1))  # crude high-pass
    return np.asarray(0.08 * noise * np.exp(-np.arange(n) / sr * 80), dtype=np.float64)


def _tone(midi: int, seconds: float, sr: int, amp: float, harmonics: int = 3) -> FloatArray:
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * _hz(midi) * h * t) / h for h in range(1, harmonics + 1))
    envelope = np.minimum(1.0, t / 0.01) * np.minimum(1.0, (seconds - t) / 0.05)
    return np.asarray(amp * wave * envelope, dtype=np.float64)


def render_track(
    path: Path,
    *,
    bpm: float = 124.0,
    intro_bars: int = 8,
    main_bars: int = 24,
    outro_bars: int = 16,
    sr: int = 22050,
    gain: float = 1.0,
    busy: bool = True,
    seed: int = 0,
) -> float:
    """Write a WAV and return its duration in seconds.

    Intro and outro: kick and hats. Main: kick, hats, offbeat bass and chords.
    busy=False drops the hats and halves the kicks (a sparse, low-energy track).
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bars = intro_bars + main_bars + outro_bars
    duration = bars * 4 * beat + 1.0
    out = np.zeros(int(duration * sr))
    kick, hat = _kick(sr), _hat(sr, rng)

    for bar in range(bars):
        in_main = intro_bars <= bar < intro_bars + main_bars
        chord_i = (bar // 2) % len(A_MINOR_PROGRESSION)
        bar_start = bar * 4 * beat
        if in_main:
            for note in A_MINOR_PROGRESSION[chord_i]:
                _add(out, bar_start, _tone(note, 4 * beat, sr, 0.06), sr)
        for b in range(4):
            t = bar_start + b * beat
            if busy or b % 2 == 0:
                _add(out, t, kick, sr)
            if busy:
                _add(out, t + beat / 2, hat, sr)
            if in_main:
                _add(out, t + beat / 2, _tone(A_MINOR_BASS[chord_i], beat / 2, sr, 0.35, 2), sr)

    out *= gain / max(1e-9, float(np.max(np.abs(out))))
    out *= 0.9
    sf.write(path, out.astype(np.float32), sr, subtype="PCM_16")
    return duration

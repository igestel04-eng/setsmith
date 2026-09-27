"""Track, CuePoint and TempoMarker dataclasses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from setsmith.keys.camelot import parse_key
from setsmith.scoring.weights import DEFAULT_CONFIG, EnergyConfig, VocalTagConfig

# Matches energy tags in Comments: "E7", "e 7", "Energy 7", and Mixed In Key's "8A - Energy 7".
_ENERGY_TAG_RE = re.compile(r"\b(?:energy|e)\s*(10|[1-9])\b", re.IGNORECASE)


class CueType(IntEnum):
    CUE = 0
    FADE_IN = 1
    FADE_OUT = 2
    LOAD = 3
    LOOP = 4


MEMORY_CUE_NUM = -1


class EnergySource(StrEnum):
    COMMENT_TAG = "comment_tag"
    RATING = "rating"
    AUDIO = "audio"  # Phase 3


@dataclass(frozen=True, slots=True)
class TempoMarker:
    inizio_s: float  # beat position in seconds
    bpm: float
    metro: str  # time signature, for example "4/4"
    battito: int  # beat number within the bar, 1-4


@dataclass(frozen=True, slots=True)
class CuePoint:
    name: str
    type: CueType
    start_s: float
    end_s: float | None
    num: int  # -1 = memory cue, 0 and up = hot cue A, B, ...

    @property
    def is_memory_cue(self) -> bool:
        return self.num == MEMORY_CUE_NUM


@dataclass(slots=True)
class Track:
    id: str
    title: str = ""
    artist: str = ""
    remixer: str = ""
    label: str = ""
    genre: str = ""
    comments: str = ""
    location: str = ""  # file URI, exactly as read
    duration_s: float = 0.0
    bpm: float | None = None
    key_raw: str | None = None
    camelot: str | None = None
    key_confidence: float = DEFAULT_CONFIG.harmonic.default_tag_confidence
    rating: int = 0  # 0 = unrated, 1-5 stars
    colour: str | None = None
    beat_grid: list[TempoMarker] = field(default_factory=list)
    cues: list[CuePoint] = field(default_factory=list)
    energy: float | None = None
    energy_source: EnergySource | None = None
    detected_camelot: str | None = None  # Phase 3: key detected from the audio
    my_tags: list[str] = field(default_factory=list)  # Rekordbox My Tags, "Category: Tag"
    vocal: bool | None = None
    intro_bars: int | None = None
    outro_bars: int | None = None
    variable_tempo: bool = False

    @property
    def display(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.title or self.artist or f"Track {self.id}"

    def summary(self) -> dict[str, Any]:
        """Compact JSON-friendly description used by CLI output."""
        key = parse_key(self.camelot)
        return {
            "id": self.id,
            "artist": self.artist,
            "title": self.title,
            "bpm": self.bpm,
            "key": self.camelot,
            "key_name": key.name if key else None,
            "key_raw": self.key_raw,
            "key_confidence": self.key_confidence,
            "detected_key": self.detected_camelot,
            "energy": self.energy,
            "energy_source": self.energy_source.value if self.energy_source else None,
            "genre": self.genre,
            "rating": self.rating,
            "vocal": self.vocal,
            "my_tags": self.my_tags,
            "intro_bars": self.intro_bars,
            "outro_bars": self.outro_bars,
            "duration_s": self.duration_s,
            "location": self.location,
        }


def energy_from_comments(comments: str, cfg: EnergyConfig = DEFAULT_CONFIG.energy) -> float | None:
    match = _ENERGY_TAG_RE.search(comments)
    if not match:
        return None
    value = float(match.group(1))
    return value if cfg.scale_min <= value <= cfg.scale_max else None


def derive_energy(
    comments: str, rating: int, cfg: EnergyConfig = DEFAULT_CONFIG.energy
) -> tuple[float | None, EnergySource | None]:
    """Pre-audio-analysis energy: a comment tag wins, then rating, else unknown."""
    tagged = energy_from_comments(comments, cfg)
    if tagged is not None:
        return tagged, EnergySource.COMMENT_TAG
    if rating > 0:
        value = min(cfg.scale_max, max(cfg.scale_min, rating * cfg.from_rating_multiplier))
        return value, EnergySource.RATING
    return None, None


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def derive_vocal(*texts: str, cfg: VocalTagConfig = DEFAULT_CONFIG.vocal) -> bool | None:
    """True for "Vocal Mix" / a "vocal" comment, False for "Dub" / "Instrumental", else None."""
    text = " ".join(texts).casefold()
    vocal = _has_word(text, cfg.vocal_words)
    instrumental = _has_word(text, cfg.instrumental_words)
    if vocal == instrumental:
        return None
    return vocal

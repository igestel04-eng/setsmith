"""DJ style profiles: JSON files describing a style's musical parameters.

Profiles are *inspired by* a DJ's public output (track metadata, press descriptions). They
describe tempo, keys, genres and mixing habits in our own words, imply no endorsement,
and are meant to be edited.

Lookup order for `load_style("name")`: $SETSMITH_STYLES_DIR, then the user directory
($XDG_CONFIG_HOME/setsmith/styles or ~/.config/setsmith/styles), then the built-in
profiles shipped in setsmith/styles/profiles. A path to a .json file also works.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from setsmith.keys.camelot import KeyMove, parse_key
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

if TYPE_CHECKING:
    from setsmith.sets.curves import EnergyCurve

BUILTIN_DIR = Path(__file__).parent / "profiles"
TransitionKind = Literal["long_blend", "cut", "echo_out", "filter_sweep", "drop_swap"]
LengthKey = Literal["8", "16", "32", "64"]


class StyleError(ValueError):
    """A style profile could not be found or is invalid."""


class KeyModePreference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    minor: float = Field(ge=0, le=1)
    major: float = Field(ge=0, le=1)


class StyleProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    bpm_band: tuple[float, float]
    bpm_preferred: float
    max_tempo_drift_bpm: float = Field(gt=0)
    key_mode_preference: KeyModePreference
    allowed_key_moves: dict[KeyMove, float]
    energy_curve: str | list[tuple[float, float]]
    genre_weights: dict[str, float]
    vocal_density: float = Field(ge=0, le=1)
    transition_length_bars: dict[LengthKey, float]
    transition_type_mix: dict[TransitionKind, float]
    reference_artists: list[str] = []
    reference_labels: list[str] = []
    notes: str = ""
    sources: list[str] = []

    _genres: dict[str, float] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> StyleProfile:
        lo, hi = self.bpm_band
        if not 0 < lo < hi:
            raise ValueError(f"bpm_band must be [min, max] with 0 < min < max, got {self.bpm_band}")
        if KeyMove.CLASH in self.allowed_key_moves:
            raise ValueError("allowed_key_moves cannot include 'clash'")
        tolerance = DEFAULT_CONFIG.style.mix_tolerance
        for field, mix in (
            ("transition_length_bars", self.transition_length_bars),
            ("transition_type_mix", self.transition_type_mix),
        ):
            values = list(mix.values())
            if any(not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > tolerance:
                raise ValueError(f"{field} shares must be 0-1 and sum to 1, got {sum(values):.2f}")
        for field, weights in (
            ("allowed_key_moves", self.allowed_key_moves),
            ("genre_weights", self.genre_weights),
        ):
            if any(not 0 <= w <= 1 for w in weights.values()):
                raise ValueError(f"{field} weights must be between 0 and 1")
        return self

    def model_post_init(self, context: object, /) -> None:
        normalize = DEFAULT_CONFIG.genre.normalize
        genres: dict[str, float] = {}
        for raw, weight in self.genre_weights.items():
            for name in normalize(raw):
                genres[name] = max(weight, genres.get(name, 0.0))
        self._genres = genres

    # ------------------------------------------------------------ scoring helpers

    def genre_weight(self, genre: str, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
        names = cfg.genre.normalize(genre)
        if not names:
            return cfg.style.unknown_genre_weight
        return max(self._genres.get(n, cfg.style.unlisted_genre_weight) for n in names)

    def key_move_weight(self, move: KeyMove) -> float:
        return self.allowed_key_moves.get(move, 0.0)

    def type_share(self, kind: str) -> float:
        return float(self.transition_type_mix.get(kind, 0.0))  # type: ignore[call-overload]

    def prefers_long_blends(self, cfg: ScoringConfig = DEFAULT_CONFIG) -> bool:
        return self.transition_length_bars.get("64", 0.0) >= cfg.style.long_blend_64_min_share

    def bpm_fit(self, bpm: float | None, cfg: ScoringConfig = DEFAULT_CONFIG) -> float | None:
        if not bpm:
            return None
        lo, hi = self.bpm_band
        outside = max(lo - bpm, bpm - hi, 0.0)
        return max(0.0, 1.0 - outside / cfg.style.bpm_falloff_bpm)

    def style_fit(self, track: Track, cfg: ScoringConfig = DEFAULT_CONFIG) -> StyleFit:
        """Weighted average of BPM band, genre weight, key mode and vocal parts (0-1)."""
        key = parse_key(track.camelot)
        parts: dict[str, float] = {}
        if (bpm := self.bpm_fit(track.bpm, cfg)) is not None:
            parts["bpm"] = bpm
        if track.genre:
            parts["genre"] = self.genre_weight(track.genre, cfg)
        if key is not None:
            pref = self.key_mode_preference
            parts["key_mode"] = pref.minor if key.is_minor else pref.major
        if track.vocal is not None:
            parts["vocal"] = self.vocal_density if track.vocal else 1 - self.vocal_density
        weights = cfg.style.fit_weights
        total_weight = sum(weights[p] for p in parts)
        score = sum(weights[p] * v for p, v in parts.items()) / total_weight if parts else 0.5
        return StyleFit(score, parts)

    def curve(self, cfg: ScoringConfig = DEFAULT_CONFIG) -> EnergyCurve:
        from setsmith.sets.curves import parse_curve

        if isinstance(self.energy_curve, str):
            return parse_curve(self.energy_curve, cfg)
        return parse_curve(",".join(f"{p}:{e}" for p, e in self.energy_curve), cfg)


@dataclass(frozen=True, slots=True)
class StyleFit:
    score: float  # 0-1
    parts: dict[str, float]

    def describe(self) -> str:
        return ", ".join(f"{k} {v:.2f}" for k, v in self.parts.items()) or "no data"


# ---------------------------------------------------------------- loading


def user_styles_dir() -> Path:
    if env := os.environ.get("SETSMITH_STYLES_DIR"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "setsmith" / "styles"


@dataclass(frozen=True, slots=True)
class StyleEntry:
    key: str  # file stem, used with --style
    path: Path
    builtin: bool


def list_styles() -> list[StyleEntry]:
    """Available profiles; a user profile hides a built-in one with the same name."""
    found: dict[str, StyleEntry] = {}
    for directory, builtin in ((BUILTIN_DIR, True), (user_styles_dir(), False)):
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                found[path.stem] = StyleEntry(path.stem, path, builtin)
    return sorted(found.values(), key=lambda e: e.key)


def load_style_file(path: Path) -> StyleProfile:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StyleError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise StyleError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        return StyleProfile.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'profile'}: {err['msg']}"
            for err in exc.errors()
        )
        raise StyleError(f"{path.name} is not a valid style profile: {problems}") from exc


def load_style(name_or_path: str) -> StyleProfile:
    """Load a profile by name (see list_styles) or by path to a .json file."""
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix == ".json" or candidate.is_file():
        return load_style_file(candidate)
    key = name_or_path.strip().lower().replace(" ", "_").replace("-", "_")
    for entry in list_styles():
        if entry.key == key:
            return load_style_file(entry.path)
    names = ", ".join(e.key for e in list_styles()) or "none"
    raise StyleError(f"no style profile named {name_or_path!r} (available: {names})")

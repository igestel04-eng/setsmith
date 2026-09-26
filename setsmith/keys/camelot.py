"""Key parsing, normalization to Camelot, and Camelot move classification.

Accepted input notations (case-insensitive, surrounding whitespace ignored):

- Camelot: ``8A``, ``12B``
- Open Key: ``1m``, ``6d`` (``m`` = minor, ``d`` = major)
- Classic: ``Am``, ``F#m``, ``Abm``, ``C``, ``Db``, ``A minor``, ``Eb maj``, ``G♯m``

Enharmonic spellings resolve to the same Camelot key (``G#m`` == ``Abm`` == ``1A``).
Parsing never raises: unrecognized input returns ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Literal

Letter = Literal["A", "B"]

CAMELOT_WHEEL_SIZE = 12
# Camelot number of Open Key 1m/1d. Open Key = ((Camelot - 8) mod 12) + 1.
OPEN_KEY_ORIGIN = 8

# Camelot number -> (minor key name, major key name). Ground truth from the spec.
CAMELOT_TABLE: dict[int, tuple[str, str]] = {
    1: ("Ab minor", "B major"),
    2: ("Eb minor", "F# major"),
    3: ("Bb minor", "Db major"),
    4: ("F minor", "Ab major"),
    5: ("C minor", "Eb major"),
    6: ("G minor", "Bb major"),
    7: ("D minor", "F major"),
    8: ("A minor", "C major"),
    9: ("E minor", "G major"),
    10: ("B minor", "D major"),
    11: ("F# minor", "A major"),
    12: ("Db minor", "E major"),
}

_NATURAL_PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_MINOR_WORDS = {"m", "min", "minor"}
_MAJOR_WORDS = {"", "maj", "major"}

_CAMELOT_RE = re.compile(r"^(\d{1,2})\s*([AB])$", re.IGNORECASE)
_OPEN_KEY_RE = re.compile(r"^(\d{1,2})\s*([MD])$", re.IGNORECASE)
_CLASSIC_RE = re.compile(r"^([A-G])\s*([#♯b♭]?)\s*(minor|major|min|maj|m)?$", re.IGNORECASE)


def _pitch_class(name: str) -> int:
    """Pitch class (C=0) of a note name such as 'F#' or 'Db'."""
    return (_NATURAL_PITCH[name[0].upper()] + _ACCIDENTAL[name[1:]]) % CAMELOT_WHEEL_SIZE


# (pitch class, is_minor) -> Camelot number, derived from CAMELOT_TABLE.
_PITCH_TO_CAMELOT: dict[tuple[int, bool], int] = {}
for _number, (_minor_name, _major_name) in CAMELOT_TABLE.items():
    _PITCH_TO_CAMELOT[(_pitch_class(_minor_name.split()[0]), True)] = _number
    _PITCH_TO_CAMELOT[(_pitch_class(_major_name.split()[0]), False)] = _number


@dataclass(frozen=True, slots=True)
class CamelotKey:
    number: int
    letter: Letter

    def __post_init__(self) -> None:
        if not 1 <= self.number <= CAMELOT_WHEEL_SIZE:
            raise ValueError(f"Camelot number out of range: {self.number}")
        if self.letter not in ("A", "B"):
            raise ValueError(f"Camelot letter must be A or B: {self.letter!r}")

    def __str__(self) -> str:
        return f"{self.number}{self.letter}"

    @property
    def is_minor(self) -> bool:
        return self.letter == "A"

    @property
    def name(self) -> str:
        """Classic key name, for example 'A minor'."""
        minor, major = CAMELOT_TABLE[self.number]
        return minor if self.is_minor else major

    @property
    def short_name(self) -> str:
        """Compact classic name as Rekordbox writes it, for example 'Am' or 'C'."""
        note = self.name.split()[0]
        return f"{note}m" if self.is_minor else note

    @property
    def open_key(self) -> str:
        number = (self.number - OPEN_KEY_ORIGIN) % CAMELOT_WHEEL_SIZE + 1
        return f"{number}{'m' if self.is_minor else 'd'}"


@lru_cache(maxsize=1024)
def parse_key(raw: str | None) -> CamelotKey | None:
    """Parse a key in Camelot, Open Key, or classic notation. Returns None if unrecognized."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    if match := _CAMELOT_RE.match(text):
        number = int(match.group(1))
        if 1 <= number <= CAMELOT_WHEEL_SIZE:
            return CamelotKey(number, "A" if match.group(2).upper() == "A" else "B")
        return None

    if match := _OPEN_KEY_RE.match(text):
        open_number = int(match.group(1))
        if 1 <= open_number <= CAMELOT_WHEEL_SIZE:
            number = (open_number - 1 + OPEN_KEY_ORIGIN - 1) % CAMELOT_WHEEL_SIZE + 1
            return CamelotKey(number, "A" if match.group(2).lower() == "m" else "B")
        return None

    if match := _CLASSIC_RE.match(text):
        note, accidental, mode = match.group(1), match.group(2), (match.group(3) or "")
        # Only a lowercase 'b' is a flat; 'B' after a note letter is not valid classic notation.
        if accidental == "B":
            return None
        mode = mode.lower()
        if mode in _MINOR_WORDS:
            is_minor = True
        elif mode in _MAJOR_WORDS:
            is_minor = False
        else:
            return None
        pitch = _pitch_class(note + accidental.replace("♯", "#").replace("♭", "b"))
        number = _PITCH_TO_CAMELOT[(pitch, is_minor)]
        return CamelotKey(number, "A" if is_minor else "B")

    return None


def normalize_key(raw: str | None) -> str | None:
    """Normalize any supported key notation to a Camelot string such as '8A'."""
    key = parse_key(raw)
    return str(key) if key else None


def all_keys() -> list[CamelotKey]:
    """All 24 Camelot keys, 1A..12A then 1B..12B."""
    letters: tuple[Letter, Letter] = ("A", "B")
    return [CamelotKey(n, letter) for letter in letters for n in range(1, CAMELOT_WHEEL_SIZE + 1)]


class KeyMove(StrEnum):
    """Harmonic relationship of a transition from key a to key b."""

    SAME = "same"
    ADJACENT = "adjacent"  # +/-1, same letter
    RELATIVE = "relative"  # A/B swap, same number
    ENERGY_BOOST = "energy_boost"  # +2, same letter
    DIAGONAL = "diagonal"  # +/-1 and letter swap
    SEMITONE_LIFT = "semitone_lift"  # +7 (== -5), same letter: one semitone up
    CLASH = "clash"


_STEP_ADJACENT = {1, CAMELOT_WHEEL_SIZE - 1}
_STEP_ENERGY_BOOST = 2
_STEP_SEMITONE_UP = 7


def camelot_step(a: CamelotKey, b: CamelotKey) -> int:
    """Signed wheel distance from a to b in the range -5..6."""
    step = (b.number - a.number) % CAMELOT_WHEEL_SIZE
    half = CAMELOT_WHEEL_SIZE // 2
    return step - CAMELOT_WHEEL_SIZE if step > half else step


def classify_move(a: CamelotKey, b: CamelotKey) -> KeyMove:
    """Classify the Camelot move from a to b. Direction matters (+2 is a boost, -2 is not)."""
    step = (b.number - a.number) % CAMELOT_WHEEL_SIZE
    if a.letter == b.letter:
        if step == 0:
            return KeyMove.SAME
        if step in _STEP_ADJACENT:
            return KeyMove.ADJACENT
        if step == _STEP_ENERGY_BOOST:
            return KeyMove.ENERGY_BOOST
        if step == _STEP_SEMITONE_UP:
            return KeyMove.SEMITONE_LIFT
        return KeyMove.CLASH
    if step == 0:
        return KeyMove.RELATIVE
    if step in _STEP_ADJACENT:
        return KeyMove.DIAGONAL
    return KeyMove.CLASH

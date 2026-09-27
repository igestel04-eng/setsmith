"""Parse user-supplied tracklists and match them to the collection.

Accepts pasted text in the common shapes, one track per line:

    01. Artist - Title
    [00:03:15] Artist - Title (Some Remix) [Label]
    1:02:33 Artist - Title       (en and em dashes work too)
    w/ Artist - Title          (played together with the previous line)
    ID - ID                    (unidentified)

and CSV with a header row naming artist/title columns (and optionally time/start).

Setsmith never scrapes tracklist sites. `TracklistProvider` is the extension point for a
licensed tracklist source.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

_SECONDS_PER_MINUTE = 60
_TIME_RE = r"(?:(\d{1,2}):)?(\d{1,3}):(\d{2})"
_LINE_RE = re.compile(
    rf"""^\s*
    (?:\#?(?P<index>\d{{1,3}})[.)\]:]?\s+)?          # 01.  1)  #1
    (?:[\[(]?(?P<time>{_TIME_RE})[\])]?\s*[-\u2013\u2014|:]?\s*)?  # [00:03:15]  1:02:33 -
    (?P<together>w/\s*)?
    (?P<body>.+?)\s*$""",
    re.VERBOSE,
)
_TIME_ONLY_RE = re.compile(rf"^{_TIME_RE}$")
_SEPARATORS = (" - ", " \u2013 ", " \u2014 ", " -- ")  # hyphen, en dash, em dash
_LABEL_RE = re.compile(r"\s*\[(?P<label>[^\]]+)\]\s*$")
_UNKNOWN = {"id", "id - id", "?", "unknown", "unknown - unknown"}
_CSV_ARTIST = ("artist", "artists")
_CSV_TITLE = ("title", "track", "name", "song")
_CSV_TIME = ("time", "start", "timestamp", "cue")


@dataclass(frozen=True, slots=True)
class TracklistEntry:
    position: int  # 1-based, in play order
    raw: str
    artist: str
    title: str
    start_s: float | None = None
    label: str = ""
    unknown: bool = False  # "ID - ID"
    together: bool = False  # "w/": layered with the previous entry

    @property
    def query(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TracklistProvider(Protocol):
    """Extension point for a licensed tracklist source (not implemented here)."""

    def tracklist(self, set_reference: str) -> list[TracklistEntry]: ...


def parse_time(text: str) -> float | None:
    match = _TIME_ONLY_RE.match(text.strip())
    if not match:
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if seconds >= _SECONDS_PER_MINUTE:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _split_body(body: str) -> tuple[str, str, str]:
    """(artist, title, label) from "Artist - Title [Label]"."""
    label = ""
    if (m := _LABEL_RE.search(body)) and any(sep in body[: m.start()] for sep in _SEPARATORS):
        label = m.group("label").strip()
        body = body[: m.start()]
    for sep in _SEPARATORS:
        if sep in body:
            artist, title = body.split(sep, 1)
            return artist.strip(), title.strip(), label
    return "", body.strip(), label


def parse_text(text: str) -> list[TracklistEntry]:
    entries: list[TracklistEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#!", "//")) or set(stripped) <= set("-=*_ "):
            continue
        match = _LINE_RE.match(stripped)
        if not match:
            continue
        body = match.group("body")
        artist, title, label = _split_body(body)
        numbered = match.group("index") or match.group("time")
        if not title or (not artist and not numbered):  # headers like "Tracklist:"
            continue
        time_text = match.group("time")
        unknown = body.strip().casefold() in _UNKNOWN or (
            artist.casefold() == "id" and title.casefold() == "id"
        )
        entries.append(
            TracklistEntry(
                position=len(entries) + 1,
                raw=stripped,
                artist=artist,
                title=title,
                start_s=parse_time(time_text) if time_text else None,
                label=label,
                unknown=unknown,
                together=bool(match.group("together")),
            )
        )
    return entries


def _find_column(header: list[str], names: tuple[str, ...]) -> int | None:
    lowered = [h.strip().casefold() for h in header]
    return next((i for i, h in enumerate(lowered) if h in names), None)


def parse_csv(text: str) -> list[TracklistEntry] | None:
    """Entries from CSV with a header naming a title column; None if it isn't such a CSV."""
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        return None
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return None
    header = rows[0]
    title_col = _find_column(header, _CSV_TITLE)
    if title_col is None:
        return None
    artist_col = _find_column(header, _CSV_ARTIST)
    time_col = _find_column(header, _CSV_TIME)
    entries: list[TracklistEntry] = []
    for row in rows[1:]:
        if len(row) <= title_col or not row[title_col].strip():
            continue
        artist = row[artist_col].strip() if artist_col is not None and artist_col < len(row) else ""
        title = row[title_col].strip()
        start = parse_time(row[time_col]) if time_col is not None and time_col < len(row) else None
        entries.append(
            TracklistEntry(
                position=len(entries) + 1,
                raw=",".join(row),
                artist=artist,
                title=title,
                start_s=start,
                unknown=artist.casefold() == "id" and title.casefold() == "id",
            )
        )
    return entries


def parse_tracklist(text: str) -> list[TracklistEntry]:
    """CSV when the first line is a header naming a title column, else free text."""
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if any(d in first for d in ",;\t") and any(n in first.casefold() for n in _CSV_TITLE):
        parsed = parse_csv(text)
        if parsed is not None:
            return parsed
    return parse_text(text)


def load_tracklist(path: Path) -> list[TracklistEntry]:
    return parse_tracklist(path.read_text(encoding="utf-8-sig"))


def timestamps_usable(entries: list[TracklistEntry]) -> bool:
    """True when every entry has a start time and they only go forward."""
    times = [e.start_s for e in entries]
    if not times or any(t is None for t in times):
        return False
    known = [t for t in times if t is not None]
    return all(b > a for a, b in pairwise(known))


# ---------------------------------------------------------------- matching


@dataclass(frozen=True, slots=True)
class TracklistMatch:
    entry: TracklistEntry
    track: Track | None
    score: float
    ambiguous: bool = False  # several collection entries (e.g. duplicates) fit equally

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "track_id": self.track.id if self.track else None,
            "track": self.track.display if self.track else None,
            "score": round(self.score, 3),
            "ambiguous": self.ambiguous,
        }


def match_tracklist(
    collection: Collection, entries: list[TracklistEntry], cfg: ScoringConfig = DEFAULT_CONFIG
) -> list[TracklistMatch]:
    matches = []
    for entry in entries:
        if entry.unknown:
            matches.append(TracklistMatch(entry, None, 0.0))
            continue
        found = collection.find(entry.query, cfg.search)
        if found.track is not None:
            matches.append(TracklistMatch(entry, found.track, found.candidates[0][0]))
        elif found.candidates and found.candidates[0][0] >= cfg.search.confident_match:
            score, track = found.candidates[0]
            matches.append(TracklistMatch(entry, track, score, ambiguous=True))
        else:
            best = found.candidates[0][0] if found.candidates else 0.0
            matches.append(TracklistMatch(entry, None, best))
    return matches

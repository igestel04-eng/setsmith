"""A loaded collection: tracks, playlist tree, and track lookup by "Artist - Title"."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import IntEnum
from pathlib import Path

from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG, SearchConfig


class NodeType(IntEnum):
    FOLDER = 0
    PLAYLIST = 1


class KeyType(IntEnum):
    TRACK_ID = 0
    LOCATION = 1


@dataclass(slots=True)
class PlaylistNode:
    name: str
    type: NodeType
    key_type: KeyType = KeyType.TRACK_ID
    children: list[PlaylistNode] = field(default_factory=list)
    track_keys: list[str] = field(default_factory=list)  # TrackIDs or Locations per key_type

    def walk(self, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], PlaylistNode]]:
        here = (*path, self.name)
        yield here, self
        for child in self.children:
            yield from child.walk(here)


@dataclass(slots=True)
class Collection:
    tracks: dict[str, Track]
    playlists: PlaylistNode | None = None
    source_path: Path | None = None
    product: str = ""
    warnings: list[str] = field(default_factory=list)
    _names: dict[str, tuple[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.tracks)

    def playlist_tracks(self, node: PlaylistNode) -> list[Track]:
        if node.key_type == KeyType.TRACK_ID:
            return [self.tracks[k] for k in node.track_keys if k in self.tracks]
        by_location = {t.location: t for t in self.tracks.values()}
        return [by_location[k] for k in node.track_keys if k in by_location]

    def normalized_names(self, track: Track) -> tuple[str, str]:
        """(artist, title) run through normalize_text, cached per TrackID."""
        names = self._names.get(track.id)
        if names is None:
            names = (normalize_text(track.artist), strip_versions(normalize_text(track.title)))
            self._names[track.id] = names
        return names

    def find(self, query: str, cfg: SearchConfig = DEFAULT_CONFIG.search) -> TrackMatch:
        return find_track(self, query, cfg)


@dataclass(slots=True)
class TrackMatch:
    """Result of a track lookup. `track` is set only when the match is unambiguous."""

    query: str
    track: Track | None
    candidates: list[tuple[float, Track]]

    @property
    def ambiguous(self) -> bool:
        return self.track is None and bool(self.candidates)


_PUNCT_RE = re.compile(r"[^\w\s&]+")
_SPACE_RE = re.compile(r"\s+")
_ARTIST_TITLE_SEP = " - "


def normalize_text(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = _PUNCT_RE.sub(" ", no_accents.casefold())
    return _SPACE_RE.sub(" ", cleaned).strip()


def strip_versions(normalized: str, cfg: SearchConfig = DEFAULT_CONFIG.search) -> str:
    """Drop version labels like "extended mix" from normalized text (whole words only)."""
    text = f" {normalized} "
    for phrase in cfg.version_noise:
        text = text.replace(f" {phrase} ", " ")
    return " ".join(text.split()) or normalized


class _Similarity:
    """Similarity of one fixed query against many field values.

    The query is difflib's cached second sequence, so its index is built once per search.
    """

    def __init__(self, query: str, cfg: SearchConfig) -> None:
        self.query = query
        self.cfg = cfg
        self.matcher = SequenceMatcher(None, autojunk=False)
        self.matcher.set_seq2(query)

    def __call__(self, value: str, *, bound: bool = False) -> float:
        """Similarity 0-1. With bound=True, a cheap upper bound of the fuzzy part instead."""
        query, cfg = self.query, self.cfg
        if not query:
            return 1.0
        if not value:
            return 0.0
        if query == value:
            return 1.0
        if query in value:
            # "move" inside "move extended mix": high, but below an exact match.
            return cfg.substring_base + (1.0 - cfg.substring_base) * len(query) / len(value)
        self.matcher.set_seq1(value)
        return self.matcher.quick_ratio() if bound else self.matcher.ratio()


def find_track(
    collection: Collection, query: str, cfg: SearchConfig = DEFAULT_CONFIG.search
) -> TrackMatch:
    """Find a track by TrackID, "Artist - Title", or title alone.

    Returns an unambiguous match only when one track clearly beats the rest. Duplicate
    entries of the same song (same normalized artist and title) count as ambiguous so the
    caller can choose by TrackID.
    """
    if query in collection.tracks:
        return TrackMatch(query, collection.tracks[query], [(1.0, collection.tracks[query])])

    if _ARTIST_TITLE_SEP in query:
        artist_q, title_q = (normalize_text(p) for p in query.split(_ARTIST_TITLE_SEP, 1))
    else:
        artist_q, title_q = "", normalize_text(query)
    title_q = strip_versions(title_q, cfg)

    title_sim, artist_sim = _Similarity(title_q, cfg), _Similarity(artist_q, cfg)

    def combined(artist: str, title: str, bound: bool) -> float:
        title_score = title_sim(title, bound=bound)
        if not artist_q:
            return title_score
        artist_score = artist_sim(artist, bound=bound)
        return cfg.artist_share * artist_score + (1 - cfg.artist_share) * title_score

    scored: list[tuple[float, Track]] = []
    for track in collection.tracks.values():
        artist, title = collection.normalized_names(track)
        if combined(artist, title, bound=True) < cfg.min_match:
            continue
        score = combined(artist, title, bound=False)
        if score >= cfg.min_match:
            scored.append((score, track))

    scored.sort(key=lambda st: (-st[0], st[1].display))
    candidates = scored[: cfg.max_candidates]
    if not candidates:
        return TrackMatch(query, None, [])

    best_score, best = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    unique_exact = best_score >= 1.0 > runner_up
    if unique_exact or (
        best_score >= cfg.confident_match and best_score - runner_up >= cfg.min_gap
    ):
        return TrackMatch(query, best, candidates)
    return TrackMatch(query, None, candidates)

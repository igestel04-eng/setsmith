"""Any song as a seed or opening track, in the collection or not.

A song in the collection is used as is (with Rekordbox's BPM and key). Otherwise it is
looked up: "Artist - Title" is taken literally, free text is searched on Last.fm, BPM and
key come from GetSongBPM or the Deezer preview, and the genre from Last.fm's artist tags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from setsmith.discovery.discover import (
    artist_genre,
    artists_match,
    identity,
    listen_links,
    make_clients,
)
from setsmith.discovery.lastfm import LastFm
from setsmith.discovery.tempokey import lookup_tempo_key
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.store import Store

EXTERNAL_PREFIX = "external:"  # IDs of tracks outside the collection
EXTERNAL_ID = f"{EXTERNAL_PREFIX}song"
_SEPARATOR = " - "


@dataclass(slots=True)
class Song:
    track: Track
    in_library: bool
    bpm_key_source: str | None = None  # None for library tracks (Rekordbox's own values)
    links: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.track.summary(),
            "in_library": self.in_library,
            "bpm_key_source": self.bpm_key_source,
            "links": self.links,
            "warnings": self.warnings,
        }


def is_external(track: Track) -> bool:
    return track.id.startswith(EXTERNAL_PREFIX)


def with_track(collection: Collection, track: Track) -> Collection:
    """A copy of the collection that also holds `track` (the original is untouched)."""
    return replace(collection, tracks={**collection.tracks, track.id: track}, warnings=[])


def library_match(
    collection: Collection, artist: str, title: str, cfg: ScoringConfig = DEFAULT_CONFIG
) -> Track | None:
    """The collection's copy of this exact song: same title (ignoring version labels) and a
    shared artist. "Move" does not match "Move Your Body". With several copies (two uploads
    of one track), the closest search match is used."""
    want_artists, want_title = identity(artist, title, packed=False)
    match = collection.find(f"{artist}{_SEPARATOR}{title}", cfg.search)
    found = [match.track] if match.track is not None else []
    found += [t for _, t in match.candidates if t is not match.track]
    for track in found:
        artists, clean = identity(track.artist, track.title)
        if clean == want_title and artists_match(want_artists, artists):
            return track
    return None


def library_ids(track_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Track IDs that can go into a Rekordbox playlist (songs outside it can't)."""
    return tuple(t for t in track_ids if not t.startswith(EXTERNAL_PREFIX))


def outside_warning(track_ids: list[str]) -> str | None:
    """A note on songs in the set that aren't in the library, if any."""
    count = sum(1 for t in track_ids if t.startswith(EXTERNAL_PREFIX))
    if not count:
        return None
    return (
        f"{count} song(s) in this set aren't in your library (marked new): get them from the "
        "links before you play. The Rekordbox export has only your own tracks, in set order"
    )


def song_outside(song: Song) -> dict[str, Any]:
    """Outside-song info (links, source) for a looked-up song."""
    t = song.track
    return {
        "artist": t.artist,
        "title": t.title,
        "found_via": "your opening song",
        "bpm_key_source": song.bpm_key_source,
        "links": song.links,
    }


def parse_song(query: str, lfm: LastFm | None) -> tuple[str, str]:
    """(artist, title) from "Artist - Title", or from a Last.fm search for free text."""
    if _SEPARATOR in query:
        artist, title = (part.strip() for part in query.split(_SEPARATOR, 1))
        if artist and title:
            return artist, title
    if lfm is None:
        raise ValueError(
            'write the song as "Artist - Title" (or save a Last.fm key to search free text)'
        )
    hits = lfm.search_track(query, 1)
    if not hits:
        raise ValueError(f'Last.fm found no song for {query!r}; try "Artist - Title"')
    return hits[0]


def resolve_song(
    query: str,
    collection: Collection,
    store: Store | None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> Song:
    """The song as a Track: from the collection when it is there, else looked up."""
    query = query.strip()
    if not query:
        raise ValueError("give a song")
    lfm, bpm, previews = make_clients(store, cfg)
    artist, title = parse_song(query, lfm)
    owned = library_match(collection, artist, title, cfg)
    if owned is not None:
        return Song(owned, in_library=True)

    infos, notes = lookup_tempo_key([(artist, title)], bpm, previews, cfg, on_progress)
    info = infos[0]
    track = Track(
        id=EXTERNAL_ID,
        artist=artist,
        title=title,
        genre=artist_genre(lfm, artist, cfg) if lfm else "",
        bpm=info.bpm if info else None,
        key_raw=info.camelot if info else None,
        camelot=info.camelot if info else None,
        key_confidence=info.key_confidence if info else cfg.harmonic.default_tag_confidence,
    )
    links = {**listen_links(artist, title, ""), **(info.links if info else {})}
    song = Song(track, in_library=False, links=links, warnings=notes)
    if track.bpm or track.camelot:
        song.bpm_key_source = info.source if info else None
    missing = [name for name, value in (("BPM", track.bpm), ("key", track.camelot)) if not value]
    if missing:
        song.warnings.append(
            f"no {' or '.join(missing)} found for {track.display}, so transitions from it "
            "are scored as unknown on that part"
        )
    if previews is None and bpm is None:
        song.warnings.append(
            "BPM and key lookups need a GetSongBPM key or the audio extra (Deezer previews)"
        )
    return song

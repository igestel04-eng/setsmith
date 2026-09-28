"""Songs outside the library for building sets, gathered around songs the DJ picks.

For each picked song (any song, owned or not) discovery finds similar songs from the same
scene that the collection doesn't have, with BPM and key from GetSongBPM or Deezer
previews. Those with a BPM join the set builder's pool; the picked songs themselves join
too when they are outside the collection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from setsmith.discovery.discover import DiscoveryUnavailable, discover, identity, make_clients
from setsmith.discovery.song import Song, resolve_song
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.store import Store

NEW_PREFIX = "external:new:"


@dataclass(slots=True)
class NewSongs:
    seeds: list[Song] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    # By track ID: where to get each song and how it was found.
    outside: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(t.id for t in self.tracks)

    def add(self, track: Track, links: dict[str, str], via: str, source: str | None) -> None:
        track = replace(track, id=f"{NEW_PREFIX}{len(self.tracks) + 1}")
        self.tracks.append(track)
        self.outside[track.id] = {
            "artist": track.artist,
            "title": track.title,
            "found_via": via,
            "bpm_key_source": source,
            "links": links,
        }


def gather_new_songs(
    queries: list[str],
    collection: Collection,
    store: Store | None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> NewSongs:
    """New songs around each query (see the module docstring)."""
    dc = cfg.discovery
    queries = [q.strip() for q in queries if q.strip()]
    if not queries:
        raise ValueError("give at least one song to find new songs around")
    if len(queries) > dc.max_seed_songs:
        raise ValueError(f"give at most {dc.max_seed_songs} songs to find new songs around")
    lfm, bpm, previews = make_clients(store, cfg)
    if lfm is None:
        raise DiscoveryUnavailable(
            "finding new songs needs a Last.fm API key: see 'setsmith keys set lastfm'"
        )

    result = NewSongs()
    seen: set[tuple[frozenset[str], str]] = set()

    def fresh(track: Track) -> bool:
        key = identity(track.artist, track.title, packed=False)
        if key in seen:
            return False
        seen.add(key)
        return True

    umbrella = set(dc.umbrella_genres)
    # Your own genre tags per artist: more specific than Last.fm's, which often say only
    # "house" for a whole scene.
    by_artist: dict[str, Counter[str]] = {}
    for owned in collection.tracks.values():
        if owned.genre and not cfg.genre.normalize(owned.genre) <= umbrella:
            for name in identity(owned.artist, owned.title)[0]:
                by_artist.setdefault(name, Counter())[owned.genre] += 1

    def specific(genre: str) -> bool:
        return not cfg.genre.normalize(genre) <= umbrella

    def library_genre(track: Track) -> str:
        votes: Counter[str] = Counter()
        for name in identity(track.artist, track.title, packed=False)[0]:
            votes.update(by_artist.get(name, Counter()))
        return votes.most_common(1)[0][0] if votes else ""

    def with_genre(track: Track, seed_genre: str) -> Track:
        """Your genre for the artist, else the picked song's, else Last.fm's."""
        genre = library_genre(track) or (
            track.genre if specific(track.genre) else seed_genre or track.genre
        )
        return replace(track, genre=genre)

    for query in queries:
        song = resolve_song(query, collection, store, cfg, on_progress)
        result.seeds.append(song)
        result.warnings += song.warnings
        seed = song.track
        fresh(seed)
        if not song.in_library:
            seed = with_genre(seed, "")
        seed_genre = seed.genre if specific(seed.genre) else ""
        if not song.in_library and seed.bpm:
            result.add(seed, song.links, "one of your picked songs", song.bpm_key_source)
        if on_progress:
            on_progress(f"finding new songs around {seed.display}")
        found = discover(
            collection, seed, lfm, bpm, top=dc.new_songs_per_seed, cfg=cfg,
            on_progress=on_progress, previews=previews,
        )  # fmt: skip
        kept = 0
        for d in found.items:
            if d.off_scene or not d.track.bpm or not fresh(d.track):
                continue
            track = with_genre(d.track, seed_genre)
            result.add(track, d.links, f"similar to {seed.display}", d.bpm_key_source)
            kept += 1
        if not kept:
            result.warnings.append(f"no new songs with a known BPM found around {seed.display}")
    return result


def with_tracks(collection: Collection, tracks: list[Track]) -> Collection:
    """A copy of the collection that also holds `tracks` (the original is untouched)."""
    added = {t.id: t for t in tracks}
    return replace(collection, tracks={**collection.tracks, **added}, warnings=[])

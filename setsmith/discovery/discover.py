"""Find tracks outside the collection that would mix well after a seed track.

1. Last.fm: tracks similar to the seed, plus top tracks of artists similar to the seed's
   artist. Anything already in the collection is dropped.
2. The best Last.fm matches (up to `max_lookups`) get BPM and key from GetSongBPM and a
   genre from their artist's Last.fm tags.
3. Each candidate is scored as a transition from the seed with the usual scorer (and
   style profile, if given). Candidates without BPM or key score neutral on those parts.

Results carry links to the track on Last.fm and searches on SoundCloud and Beatport, so
the DJ can listen and buy or add it. Nothing is downloaded.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from setsmith.discovery import getsongbpm, lastfm
from setsmith.discovery.getsongbpm import GetSongBpm
from setsmith.discovery.http import JsonClient, ServiceError
from setsmith.discovery.keys import get_key
from setsmith.discovery.lastfm import LastFm, LastFmTrack
from setsmith.model.collection import Collection, normalize_text, strip_versions
from setsmith.model.track import Track
from setsmith.scoring.transition import TransitionScore, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.sets.generate import artist_names
from setsmith.store import Store
from setsmith.styles.profile import StyleFit, StyleProfile

ATTRIBUTIONS = [lastfm.ATTRIBUTION, getsongbpm.ATTRIBUTION]
_SECONDS_PER_DAY = 86400


class DiscoveryUnavailable(RuntimeError):
    """A required API key is missing."""


@dataclass(slots=True)
class Discovery:
    track: Track
    score: TransitionScore
    style_fit: StyleFit | None
    via: str
    match: float
    links: dict[str, str]

    @property
    def has_tempo_and_key(self) -> bool:
        return bool(self.track.bpm and self.track.camelot)

    def to_dict(self) -> dict[str, Any]:
        t = self.track
        return {
            "artist": t.artist,
            "title": t.title,
            "bpm": t.bpm,
            "key": t.camelot,
            "genre": t.genre,
            "via": self.via,
            "lastfm_match": round(self.match, 3),
            "links": self.links,
            "score": self.score.to_dict(),
            "explain": self.score.explain(),
            "style_fit": {"score": round(self.style_fit.score, 3), **self.style_fit.parts}
            if self.style_fit
            else None,
        }


@dataclass(slots=True)
class DiscoveryResult:
    seed: Track
    items: list[Discovery]
    candidates: int = 0  # distinct tracks Last.fm suggested
    in_library: int = 0  # of those, already in the collection
    looked_up: int = 0
    without_tempo_key: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed.summary(),
            "candidates": self.candidates,
            "in_library": self.in_library,
            "looked_up": self.looked_up,
            "without_tempo_key": self.without_tempo_key,
            "warnings": self.warnings,
            "attribution": [{"text": text, "url": url} for text, url in ATTRIBUTIONS],
            "results": [d.to_dict() for d in self.items],
        }


class _StoreCache:
    """Adapts the Setsmith store to the HTTP client's cache interface."""

    def __init__(self, store: Store, max_entries: int) -> None:
        self.store, self.max_entries = store, max_entries

    def cache_get(self, key: str, max_age_s: float) -> str | None:
        return self.store.cache_get(key, max_age_s)

    def cache_put(self, key: str, body: str) -> None:
        self.store.cache_put(key, body, self.max_entries)


def make_sources(
    store: Store | None, cfg: ScoringConfig = DEFAULT_CONFIG
) -> tuple[LastFm, GetSongBpm | None]:
    """Clients for the configured services. Last.fm is required; GetSongBPM is optional."""
    dc = cfg.discovery
    lastfm_key = get_key("lastfm")
    if not lastfm_key:
        raise DiscoveryUnavailable(
            "a Last.fm API key is needed: create one at https://www.last.fm/api/account/create "
            "and save it with 'setsmith keys set lastfm YOUR_KEY'"
        )
    cache = _StoreCache(store, dc.cache_max_entries) if store is not None else None
    ttl = dc.cache_ttl_days * _SECONDS_PER_DAY

    def client(name: str, base: str, interval: float) -> JsonClient:
        return JsonClient(
            name,
            base,
            min_interval_s=interval,
            timeout_s=dc.timeout_s,
            cache=cache,
            cache_ttl_s=ttl,
        )

    lfm = LastFm(client("Last.fm", dc.lastfm_base, dc.lastfm_min_interval_s), lastfm_key)
    bpm_key = get_key("getsongbpm")
    bpm = (
        GetSongBpm(client("GetSongBPM", dc.getsongbpm_base, dc.getsongbpm_min_interval_s), bpm_key)
        if bpm_key
        else None
    )
    return lfm, bpm


_FEAT_RE = re.compile(r"\s+(?:feat|ft|featuring)\.?\s.*$", re.IGNORECASE)
_VERSION_PARENS_RE = re.compile(
    r"\s*[\(\[][^\)\]]*\b(?:mix|edit|extended|version|original|remaster(?:ed)?|vod)\b[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_ANY_PARENS_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_PARENS_CONTENT_RE = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
_TITLE_SEPARATOR = " - "
# Words that make up a pure version suffix like "Yamore - Edit Version" or "- 2012 Remaster".
_VERSION_WORDS = frozenset(
    {
        "original", "extended", "radio", "club", "main", "album", "single",
        "edit", "version", "mix", "remaster", "remastered", "vod",
    }
)  # fmt: skip


def _dash_version_suffix(title: str) -> bool:
    """True when the text after the last " - " is only version words (or a year)."""
    if _TITLE_SEPARATOR not in title:
        return False
    words = normalize_text(title.rsplit(_TITLE_SEPARATOR, 1)[1]).split()
    return bool(words) and all(w in _VERSION_WORDS or w.isdigit() for w in words)


def _strip_dash_version(title: str) -> str:
    return title.rsplit(_TITLE_SEPARATOR, 1)[0] if _dash_version_suffix(title) else title


_FEAT_START_RE = re.compile(r"[\(\[]\s*(?:feat|ft|featuring)\b\.?", re.IGNORECASE)
_OPEN, _CLOSE = "([", ")]"


def remove_feat_groups(title: str) -> str:
    """Drop "(feat. ...)" groups, even with brackets inside: "Yamore (feat. A, B (NL) & C)"."""
    while match := _FEAT_START_RE.search(title):
        depth, end = 0, len(title)
        for i in range(match.start(), len(title)):
            if title[i] in _OPEN:
                depth += 1
            elif title[i] in _CLOSE:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        title = (title[: match.start()] + title[end:]).strip()
    return title


def clean_title(title: str) -> str:
    """Normalized title without version labels or "feat." parts, for matching.
    Remix names stay: a remix is a different recording."""
    bare = _VERSION_PARENS_RE.sub("", remove_feat_groups(_strip_dash_version(title)))
    return _FEAT_RE.sub("", strip_versions(normalize_text(bare))).strip()


def _artists_part(text: str) -> str:
    """Drop parenthetical member lists: "Keinemusik (Rampa, &ME)" -> "Keinemusik"."""
    return _ANY_PARENS_RE.sub("", text).strip()


def _all_artists(text: str) -> set[str]:
    """Credited artists including collective members: "Keinemusik (Rampa, &ME)" gives
    keinemusik, rampa and me."""
    names = set(artist_names(_artists_part(text)))
    for inner in _PARENS_CONTENT_RE.findall(text):
        names |= artist_names(inner)
    return names


def identity(artist: str, title: str, *, packed: bool = True) -> tuple[frozenset[str], str]:
    """(individual artists, clean title) for a track. With `packed`, SoundCloud-style titles
    that carry "Artists - Title" (while the Artist field holds the uploader) are unpacked;
    Last.fm titles are never packed like that."""
    artists = _all_artists(artist)
    if packed and _TITLE_SEPARATOR in _strip_dash_version(title):
        packed_artists, packed_title = _strip_dash_version(title).split(_TITLE_SEPARATOR, 1)
        artists |= _all_artists(packed_artists)
        title = packed_title
    return frozenset(artists), clean_title(title)


def seed_query(track: Track) -> tuple[str, str]:
    """Artist and title to ask Last.fm about: the first credited artist and the bare title
    (without version, remix or feat. brackets: Last.fm knows originals best)."""
    artist, title = _artists_part(track.artist), track.title
    if _TITLE_SEPARATOR in title:
        packed_artists, title = title.split(_TITLE_SEPARATOR, 1)
        first = re.split(
            r",|&|\bx\b|\bfeat\.?|\bft\.?|\bvs\.?", _artists_part(packed_artists), maxsplit=1
        )[0]
        artist = first or artist
    title = _FEAT_RE.sub("", _ANY_PARENS_RE.sub("", remove_feat_groups(title)))
    return artist.strip(), title.strip()


def _song_key(artist: str, title: str) -> tuple[str, str]:
    return normalize_text(_artists_part(artist)), clean_title(title)


class _LibraryIndex:
    """Which songs the collection already has, tolerant of SoundCloud-style titles."""

    def __init__(self, collection: Collection) -> None:
        self.by_title: dict[str, set[str]] = {}
        for t in collection.tracks.values():
            artists, title = identity(t.artist, t.title)
            if title:
                self.by_title.setdefault(title, set()).update(artists)

    def __contains__(self, candidate: LastFmTrack) -> bool:
        artists, title = identity(candidate.artist, candidate.title, packed=False)
        owners = self.by_title.get(title)
        if not owners:
            return False
        # "moblack" matches the uploader "moblack records"
        return any(
            a == o or o.startswith(a + " ") or a.startswith(o + " ")
            for a in artists
            for o in owners
        )


def _links(artist: str, title: str, lastfm_url: str) -> dict[str, str]:
    query = urllib.parse.quote_plus(f"{artist} {title}")
    links = {
        "soundcloud": f"https://soundcloud.com/search/sounds?q={query}",
        "beatport": f"https://www.beatport.com/search?q={query}",
    }
    if lastfm_url.startswith(("https://", "http://")):  # never pass through other schemes
        links["lastfm"] = lastfm_url
    return links


def _known_genres(cfg: ScoringConfig, style: StyleProfile | None) -> set[str]:
    known = {g for group in cfg.genre.neighbor_groups for g in group}
    known |= set(cfg.genre.aliases.values())
    if style is not None:
        known |= {g for raw in style.genre_weights for g in cfg.genre.normalize(raw)}
    return known


def _genre_from_tags(tags: list[str], known: set[str], cfg: ScoringConfig) -> str:
    """The first Last.fm tag that is a dance-music genre Setsmith knows, else ""."""
    for tag in tags:
        for name in cfg.genre.normalize(tag):
            if name in known:
                return name
    return ""


def discover(
    collection: Collection,
    seed: Track,
    lfm: LastFm,
    bpm: GetSongBpm | None,
    *,
    style: StyleProfile | None = None,
    top: int = 15,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> DiscoveryResult:
    dc = cfg.discovery
    result = DiscoveryResult(seed, [])
    if not seed.artist or not seed.title:
        raise ValueError("the seed track needs an artist and a title to search Last.fm")

    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    progress("asking Last.fm for similar tracks")
    seed_artist, seed_title = seed_query(seed)
    found: list[LastFmTrack] = []
    try:
        found += lfm.similar_tracks(seed_artist, seed_title, dc.similar_tracks)
    except ServiceError as exc:
        result.warnings.append(f"no similar tracks for {seed_artist} - {seed_title} ({exc})")
    try:
        artists = lfm.similar_artists(seed_artist, dc.similar_artists)
    except ServiceError as exc:
        result.warnings.append(f"no similar artists for {seed_artist} ({exc})")
        artists = []
    for i, (artist, match) in enumerate(artists):
        progress(f"top tracks by {artist} ({i + 1}/{len(artists)})")
        try:
            found += lfm.top_tracks(artist, dc.top_tracks_per_artist, match)
        except ServiceError:
            continue

    library = _LibraryIndex(collection)
    seed_key = _song_key(seed_artist, seed_title)
    unique: dict[tuple[str, str], LastFmTrack] = {}
    for cand in sorted(found, key=lambda c: -c.match):
        key = _song_key(cand.artist, cand.title)
        if key == seed_key or key in unique:
            continue
        unique[key] = cand
    result.candidates = len(unique)
    fresh = [c for c in unique.values() if c not in library]
    result.in_library = result.candidates - len(fresh)

    known = _known_genres(cfg, style)
    tags: dict[str, str] = {}
    discoveries = []
    for i, cand in enumerate(fresh[: dc.max_lookups]):
        progress(
            f"looking up {cand.artist} - {cand.title} ({i + 1}/{min(len(fresh), dc.max_lookups)})"
        )
        if cand.artist not in tags:
            tags[cand.artist] = _genre_from_tags(lfm.artist_tags(cand.artist), known, cfg)
        info = None
        if bpm is not None:
            try:
                info = bpm.lookup(cand.artist, cand.title)
            except ServiceError as exc:
                if result.looked_up == 0:  # a bad key fails on the very first call
                    raise
                result.warnings.append(str(exc))
                bpm = None  # stop hammering a failing service
            result.looked_up += 1
        track = Track(
            id=f"discovered:{i + 1}",
            artist=cand.artist,
            title=cand.title,
            genre=tags[cand.artist],
            bpm=info.bpm if info else None,
            key_raw=info.camelot if info else None,
            camelot=info.camelot if info else None,
        )
        if not (track.bpm and track.camelot):
            result.without_tempo_key += 1
        score = score_transition(seed, track, cfg=cfg, style=style)
        links = _links(cand.artist, cand.title, cand.url)
        if info and info.url.startswith(("https://", "http://")):
            links["getsongbpm"] = info.url
        fit = style.style_fit(track, cfg) if style else None
        discoveries.append(Discovery(track, score, fit, cand.via, cand.match, links))

    if bpm is None and get_key("getsongbpm") is None:
        result.warnings.append(
            "no GetSongBPM key: results have no BPM or key, so scores lean on genre only"
        )
    # Best transitions first; known tempo/key only breaks ties (a known clash must not
    # outrank an unknown), then Last.fm's similarity.
    discoveries.sort(key=lambda d: (-round(d.score.total, 1), -d.has_tempo_and_key, -d.match))
    per_artist: dict[str, int] = {}
    varied = []
    for d in discoveries:
        who = normalize_text(d.track.artist)
        if per_artist.get(who, 0) < dc.max_per_artist:
            per_artist[who] = per_artist.get(who, 0) + 1
            varied.append(d)
    result.items = varied[:top]
    return result

"""A style profile for any artist, estimated from their most popular tracks.

Tempo range and minor/major balance come from the artist's top tracks on Last.fm, with
BPM and key from GetSongBPM or Deezer previews. Mixing habits (key moves, transition
lengths and types, energy curve) come from the genre style that matches the artist: your
own genre tags for the artist when you have any, else Last.fm's. Profiles are labeled
"inspired by", imply no endorsement, and are saved to your styles folder to edit.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from setsmith.analysis.audio import percentile
from setsmith.discovery.discover import (
    DiscoveryUnavailable,
    identity,
    make_clients,
)
from setsmith.discovery.http import ServiceError
from setsmith.discovery.tempokey import lookup_tempo_key
from setsmith.keys.camelot import parse_key
from setsmith.model.collection import Collection, normalize_text
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.store import Store
from setsmith.styles.profile import StyleProfile, list_styles, load_style_file, user_styles_dir

# Marks profiles written by this module, so re-estimating may replace them.
ESTIMATE_SOURCE = "Setsmith estimate: Last.fm top tracks, BPM/key from GetSongBPM or Deezer"


@dataclass(frozen=True, slots=True)
class ArtistStyle:
    profile: StyleProfile
    tracks: int  # popular tracks looked at
    tempos: int  # of those, with a BPM
    keys: int  # of those, with a key
    genre: str  # the artist's genre, as found
    genre_style: str  # the genre style the mixing habits come from

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.model_dump(mode="json"),
            "tracks": self.tracks,
            "tempos": self.tempos,
            "keys": self.keys,
            "genre": self.genre,
            "genre_style": self.genre_style,
        }


def _genre_styles() -> dict[str, tuple[str, StyleProfile]]:
    """Genre styles by their main genre (the one weighted 1.0)."""
    out: dict[str, tuple[str, StyleProfile]] = {}
    for entry in list_styles():
        profile = load_style_file(entry.path)
        if profile.kind == "genre":
            for genre, weight in profile.genre_weights.items():
                if weight == max(profile.genre_weights.values()):
                    out.setdefault(genre, (entry.key, profile))
    return out


def _library_genre(collection: Collection, artist: str) -> str:
    names = identity(artist, "", packed=False)[0]
    votes: Counter[str] = Counter()
    for track in collection.tracks.values():
        if track.genre and identity(track.artist, track.title)[0] & names:
            votes[track.genre] += 1
    return votes.most_common(1)[0][0] if votes else ""


def estimate_artist_style(
    artist: str,
    collection: Collection,
    store: Store | None,
    cfg: ScoringConfig = DEFAULT_CONFIG,
    on_progress: Callable[[str], None] | None = None,
) -> ArtistStyle:
    sc = cfg.style
    artist = artist.strip()
    if not artist:
        raise ValueError("give an artist")
    lfm, bpm, previews = make_clients(store, cfg)
    if lfm is None:
        raise DiscoveryUnavailable(
            "artist styles need a Last.fm API key: see 'setsmith keys set lastfm'"
        )
    if on_progress:
        on_progress(f"finding {artist}'s most popular tracks")
    try:
        top = lfm.top_tracks(artist, sc.artist_tracks, 1.0)
    except ServiceError as exc:
        raise ValueError(f"Last.fm doesn't know {artist!r} ({exc})") from exc
    if not top:
        raise ValueError(f"Last.fm has no tracks for {artist!r}")
    infos, _ = lookup_tempo_key([(t.artist, t.title) for t in top], bpm, previews, cfg, on_progress)
    tempos = sorted(i.bpm for i in infos if i and i.bpm)
    keys = [k for i in infos if i and (k := parse_key(i.camelot))]
    if len(tempos) < sc.artist_min_bpms:
        raise ValueError(
            f"found the tempo of only {len(tempos)} of {artist}'s popular tracks, too few "
            "for a style: pick a genre style instead"
        )

    # Your genre for the artist first, then Last.fm's tags, specific genres before the
    # umbrella "house" that Last.fm lists first for most dance artists; the first genre a
    # genre style covers wins.
    styles = _genre_styles()
    umbrella = set(cfg.discovery.umbrella_genres)
    tags = [g for tag in lfm.artist_tags(artist) for g in cfg.genre.normalize(tag)]
    candidates = [
        *cfg.genre.normalize(_library_genre(collection, artist)),
        *[g for g in tags if g not in umbrella],
        *[g for g in tags if g in umbrella],
    ]
    genre, genre_key, template = "", "", None
    for candidate in candidates:
        if candidate in styles:
            genre, (genre_key, template) = candidate, styles[candidate]
            break
    genre = genre or next(iter(candidates), "")
    if template is None:
        genre_key = sc.artist_fallback_genre_style
        template = next(load_style_file(e.path) for e in list_styles() if e.key == genre_key)

    step, margin = sc.artist_bpm_step, sc.artist_bpm_margin
    p_lo, p_hi = sc.artist_bpm_percentiles
    lo = math.floor((percentile(tempos, p_lo) - margin) / step) * step
    hi = math.ceil((percentile(tempos, p_hi) + margin) / step) * step
    preferred = round(statistics.median(tempos) / step) * step
    minor = (
        round(sum(1 for k in keys if k.is_minor) / len(keys), 2)
        if len(keys) >= sc.artist_min_bpms
        else template.key_mode_preference.minor
    )
    weights = dict(template.genre_weights)
    for g in cfg.genre.normalize(genre):
        weights[g] = 1.0
    try:
        similar = [a for a, _ in lfm.similar_artists(artist, sc.artist_reference_count)]
    except ServiceError:
        similar = []
    data = template.model_dump(mode="json")
    data.update(
        name=f"Inspired by {artist}",
        kind="artist",
        description=(
            f"Estimated from {len(tempos)} of {artist}'s most popular tracks: "
            f"{lo:g}-{hi:g} BPM (typically {preferred:g}), {round(100 * minor)}% minor keys. "
            f"Mixing habits are typical of {template.name}."
        ),
        bpm_band=[lo, hi],
        bpm_preferred=preferred,
        key_mode_preference={"minor": minor, "major": round(1 - minor, 2)},
        genre_weights=weights,
        reference_artists=[artist, *similar],
        reference_labels=[],
        notes="An estimate from public track metadata; edit it to match what you hear.",
        sources=[ESTIMATE_SOURCE],
    )
    profile = StyleProfile.model_validate(data)
    return ArtistStyle(profile, len(top), len(tempos), len(keys), genre, genre_key)


def save_artist_style(style: ArtistStyle) -> tuple[str, Path]:
    """Save to the user styles folder; returns (key for --style, path). Replaces an earlier
    estimate for the same artist, never a built-in or hand-made profile."""
    artist = style.profile.reference_artists[0]
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(artist)).strip("_") or "artist"
    taken = {e.key: e for e in list_styles()}
    key = slug
    entry = taken.get(key)
    if entry is not None and (
        entry.builtin or ESTIMATE_SOURCE not in load_style_file(entry.path).sources
    ):
        key = f"{slug}_estimated"
    folder = user_styles_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    text = json.dumps(style.profile.model_dump(mode="json"), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return key, path

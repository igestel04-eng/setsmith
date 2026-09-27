"""Last.fm web service (non-commercial use, attribution required: link to Last.fm pages)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from setsmith.discovery.http import JsonClient, ServiceError

ATTRIBUTION = ("Similar tracks and artists from Last.fm", "https://www.last.fm")


@dataclass(frozen=True, slots=True)
class LastFmTrack:
    artist: str
    title: str
    url: str
    match: float  # 0-1, Last.fm's similarity to the seed
    via: str  # why it was found, e.g. "similar to the seed track"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, dict) else []


class LastFm:
    def __init__(self, client: JsonClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    def _call(self, method: str, **params: str) -> dict[str, Any]:
        data = self.client.get(
            "", {"method": method, "api_key": self.api_key, "format": "json", **params}
        )
        if not isinstance(data, dict):
            raise ServiceError("Last.fm returned an unexpected response")
        if "error" in data:
            raise ServiceError(f"Last.fm: {data.get('message', 'error')} (code {data['error']})")
        return data

    def similar_tracks(self, artist: str, title: str, limit: int) -> list[LastFmTrack]:
        data = self._call(
            "track.getsimilar", artist=artist, track=title, limit=str(limit), autocorrect="1"
        )
        out = []
        for t in _as_list(data.get("similartracks", {}).get("track")):
            name, who = t.get("name"), (t.get("artist") or {}).get("name")
            if name and who:
                out.append(
                    LastFmTrack(
                        who,
                        name,
                        t.get("url", ""),
                        float(t.get("match") or 0),
                        "similar to the seed track",
                    )
                )
        return out

    def similar_artists(self, artist: str, limit: int) -> list[tuple[str, float]]:
        data = self._call("artist.getsimilar", artist=artist, limit=str(limit), autocorrect="1")
        return [
            (a["name"], float(a.get("match") or 0))
            for a in _as_list(data.get("similarartists", {}).get("artist"))
            if a.get("name")
        ]

    def top_tracks(self, artist: str, limit: int, match: float) -> list[LastFmTrack]:
        data = self._call("artist.gettoptracks", artist=artist, limit=str(limit), autocorrect="1")
        return [
            LastFmTrack(
                artist,
                t["name"],
                t.get("url", ""),
                match,
                f"popular by {artist}, similar to the seed artist",
            )
            for t in _as_list(data.get("toptracks", {}).get("track"))
            if t.get("name")
        ]

    def artist_tags(self, artist: str) -> list[str]:
        try:
            data = self._call("artist.gettoptags", artist=artist, autocorrect="1")
        except ServiceError:
            return []
        return [t["name"] for t in _as_list(data.get("toptags", {}).get("tag")) if t.get("name")]

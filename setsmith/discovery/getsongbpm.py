"""GetSongBPM API: tempo and key per song (free; a visible link to GetSongBPM.com is required)."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from setsmith.discovery.http import JsonClient, ServiceError
from setsmith.keys.camelot import normalize_key
from setsmith.model.collection import normalize_text, strip_versions

ATTRIBUTION = ("BPM and key data from GetSongBPM", "https://getsongbpm.com")
_MIN_TITLE_SIMILARITY = 0.8
_MIN_ARTIST_SIMILARITY = 0.6


@dataclass(frozen=True, slots=True)
class SongInfo:
    bpm: float | None
    camelot: str | None
    url: str


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _similar(a: str, b: str) -> float:
    a, b = strip_versions(normalize_text(a)), strip_versions(normalize_text(b))
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # A band's shorter or older name ("rufus" / "rufus du sol") counts as a match.
    if a.startswith(b + " ") or b.startswith(a + " "):
        return _MIN_ARTIST_SIMILARITY
    return SequenceMatcher(None, a, b).ratio()


def _parse_song(item: dict[str, Any]) -> SongInfo:
    key = normalize_key(item.get("open_key")) or normalize_key(item.get("key_of"))
    return SongInfo(_float(item.get("tempo")), key, str(item.get("uri") or ""))


class GetSongBpm:
    def __init__(self, client: JsonClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    def lookup(self, artist: str, title: str) -> SongInfo | None:
        """Best match for artist + title, or None. Fields may still be missing.

        Searches by title only and picks the artist from the results: the API's combined
        "song:... artist:..." lookup finds nothing once either part has several words.
        """
        song = strip_versions(normalize_text(title))
        data = self.client.get("search/", {"api_key": self.api_key, "type": "song", "lookup": song})
        if isinstance(data, dict) and "error" in data:
            message = str(data["error"])
            if "no result" in message.lower():
                return None
            raise ServiceError(f"GetSongBPM: {message}")
        results = data.get("search") if isinstance(data, dict) else None
        if isinstance(results, dict):  # {"search": {"error": "no result"}}: nothing found
            return None
        if not isinstance(results, list):
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for item in results:
            if not isinstance(item, dict):
                continue
            item_artist = (
                (item.get("artist") or {}).get("name", "")
                if isinstance(item.get("artist"), dict)
                else ""
            )
            t_sim = _similar(title, str(item.get("title", "")))
            a_sim = _similar(artist, str(item_artist))
            if t_sim >= _MIN_TITLE_SIMILARITY and a_sim >= _MIN_ARTIST_SIMILARITY:
                score = t_sim + a_sim
                if best is None or score > best[0]:
                    best = (score, item)
        if best is None:
            return None
        info = _parse_song(best[1])
        if info.bpm is None and best[1].get("id"):  # search results may be partial
            song = self.client.get("song/", {"api_key": self.api_key, "id": str(best[1]["id"])})
            if isinstance(song, dict) and isinstance(song.get("song"), dict):
                info = _parse_song(song["song"])
        return info

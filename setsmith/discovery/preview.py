"""BPM and key estimated from Deezer's official 30-second previews.

Deezer's public API (no key needed) finds the track and its preview. The preview is
decoded in memory, analyzed and discarded: only the BPM and key are kept (and cached).
Needs the `audio` extra; with Essentia installed, keys use its EDM profile and the tempo
is cross-checked by two estimators.

Estimates from 30 seconds are rougher than Rekordbox's analysis of the full track, so
keys carry a low confidence and an ambiguous tempo gives no BPM at all.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from setsmith.analysis.audio import has_module, key_from_chroma
from setsmith.discovery.http import USER_AGENT, JsonClient, ResponseCache, ServiceError
from setsmith.keys.camelot import parse_key
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

ATTRIBUTION = ("Previews for BPM and key estimates from Deezer", "https://www.deezer.com")
_SECONDS_PER_DAY = 86400
_ID3_HEADER = 10  # "ID3", version (2), flags (1), size (4 syncsafe bytes)
_ID3_FOOTER_FLAG = 0x10
_SYNCSAFE_BITS = 7


@dataclass(frozen=True, slots=True)
class DeezerHit:
    deezer_id: int
    artist: str
    title: str
    preview: str
    link: str


@dataclass(frozen=True, slots=True)
class PreviewEstimate:
    bpm: float | None
    camelot: str | None
    key_strength: float | None
    method: str  # "essentia" or "librosa"
    hit: DeezerHit

    @property
    def found(self) -> bool:
        return self.bpm is not None or self.camelot is not None


def available() -> bool:
    """Preview analysis needs the `audio` extra (librosa and soundfile)."""
    return has_module("librosa") and has_module("soundfile")


def strip_id3(data: bytes) -> bytes:
    """Drop a leading ID3v2 tag: libsndfile can't find the MP3 stream behind it in memory."""
    if len(data) < _ID3_HEADER or data[:3] != b"ID3":
        return data
    size = 0
    for byte in data[6:_ID3_HEADER]:
        size = (size << _SYNCSAFE_BITS) | (byte & 0x7F)
    size += _ID3_HEADER
    if data[5] & _ID3_FOOTER_FLAG:
        size += _ID3_HEADER
    return data[size:]


def fold_bpm(bpm: float, low: float) -> float:
    """Halve or double into [low, 2 x low): beat trackers often lock to half or double time."""
    if bpm <= 0:
        return bpm
    while bpm < low:
        bpm *= 2
    while bpm >= 2 * low:
        bpm /= 2
    return bpm


def analyze_audio(
    data: bytes, cfg: ScoringConfig = DEFAULT_CONFIG
) -> tuple[float | None, str | None, float | None, str]:
    """(BPM, Camelot key, key strength, method) for an MP3 preview held in memory."""
    import librosa
    import numpy as np
    import soundfile as sf

    dc, ac = cfg.discovery, cfg.analysis
    try:
        raw, native_sr = sf.read(io.BytesIO(strip_id3(data)), dtype="float32", always_2d=True)
    except Exception as exc:  # decoders raise many types
        raise ServiceError(f"could not decode the preview: {exc}") from None
    mono = raw.mean(axis=1)
    if len(mono) / native_sr < dc.preview_min_s:
        return None, None, None, "none"

    sr, hop = ac.sample_rate, ac.hop_length
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = librosa.resample(mono, orig_sr=native_sr, target_sr=sr)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        tempo, frames = librosa.beat.beat_track(
            onset_envelope=onset, sr=sr, hop_length=hop, start_bpm=dc.preview_start_bpm
        )
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(beats) >= dc.preview_min_beats:
        # The slope of beat times against beat number: finer than the tracker's tempo bins.
        seconds_per_beat = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
        librosa_bpm = 60.0 / seconds_per_beat if seconds_per_beat > 0 else 0.0
    else:
        librosa_bpm = float(np.atleast_1d(tempo)[0])
    librosa_bpm = fold_bpm(librosa_bpm, dc.preview_bpm_fold_min)

    if has_module("essentia"):
        import essentia
        import essentia.standard as es

        essentia.log.warningActive = False
        hi = ac.essentia_sample_rate
        y_hi = mono if native_sr == hi else librosa.resample(mono, orig_sr=native_sr, target_sr=hi)
        y_hi = np.ascontiguousarray(y_hi, dtype=np.float32)
        key, scale, strength = es.KeyExtractor(profileType=ac.essentia_key_profile, sampleRate=hi)(
            y_hi
        )
        parsed = parse_key(f"{key} {scale}")
        essentia_bpm = fold_bpm(
            float(es.RhythmExtractor2013(method="multifeature")(y_hi)[0]), dc.preview_bpm_fold_min
        )
        agree = abs(essentia_bpm - librosa_bpm) <= essentia_bpm * dc.preview_bpm_agree_pct / 100
        bpm = round(essentia_bpm, dc.preview_bpm_decimals) if agree else None
        return bpm, (str(parsed) if parsed else None), float(strength), "essentia"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr, hop_length=hop)
    camelot, corr = key_from_chroma([float(c) for c in chroma.mean(axis=1)], ac)
    bpm = round(librosa_bpm, dc.preview_bpm_decimals) if librosa_bpm > 0 else None
    return bpm, camelot, corr, "librosa"


class DeezerPreviews:
    """Finds tracks on Deezer and estimates BPM and key from their previews."""

    def __init__(
        self,
        client: JsonClient,
        cache: ResponseCache | None = None,
        cfg: ScoringConfig = DEFAULT_CONFIG,
        analyze: Callable[[bytes, ScoringConfig], tuple[Any, ...]] = analyze_audio,
        fetch: Callable[[str, ScoringConfig], bytes] | None = None,
    ) -> None:
        self.client = client
        self.cache = cache
        self.cfg = cfg
        self.analyze = analyze
        self.fetch = fetch or _fetch_preview

    def find(self, artist: str, title: str) -> DeezerHit | None:
        # Imported here: discover.py imports this module.
        from setsmith.discovery.discover import artists_match, bare_title, clean_title, identity

        data = self.client.get(
            "search",
            {
                "q": f"{artist} {bare_title(title) or title}",
                "limit": str(self.cfg.discovery.deezer_search_results),
            },
        )
        want_artists, want_title = identity(artist, title, packed=False)

        def credited(names: list[str]) -> bool:
            return any(artists_match(want_artists, identity(n, "", packed=False)[0]) for n in names)

        for item in _checked(data).get("data") or []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                continue
            names = {item.get("title") or "", item.get("title_short") or ""}
            preview, link = item.get("preview") or "", item.get("link") or ""
            if not preview or want_title not in {clean_title(n) for n in names if n}:
                continue
            who = (item.get("artist") or {}).get("name") or ""
            if not credited([who]):
                # Search results name only the main artist; featured and co-credited
                # artists are on the track itself.
                track = _checked(self.client.get(f"track/{item['id']}", {}))
                contributors = [c.get("name") or "" for c in track.get("contributors") or []]
                if not credited(contributors):
                    continue
            return DeezerHit(item["id"], who, item.get("title") or title, preview, link)
        return None

    def _cache_key(self, hit: DeezerHit) -> str:
        return f"Deezer-preview:v{self.cfg.analysis.version}:{hit.deezer_id}"

    def _estimate(self, hit: DeezerHit) -> PreviewEstimate:
        bpm, camelot, strength, method = self.analyze(self.fetch(hit.preview, self.cfg), self.cfg)
        return PreviewEstimate(bpm, camelot, strength, method, hit)

    def lookup_many(
        self,
        songs: list[tuple[str, str]],
        on_progress: Callable[[str], None] | None = None,
    ) -> tuple[list[PreviewEstimate | None], list[str]]:
        """Estimates for (artist, title) pairs, in order, and any warnings.

        Searches and cache reads run here (the cache is not thread-safe); downloads and
        analysis run in parallel threads.
        """
        dc = self.cfg.discovery
        ttl = dc.preview_cache_days * _SECONDS_PER_DAY
        results: list[PreviewEstimate | None] = [None] * len(songs)
        warnings_: list[str] = []
        todo: list[tuple[int, DeezerHit]] = []
        for i, (artist, title) in enumerate(songs):
            if on_progress:
                on_progress(f"finding {artist} - {title} on Deezer ({i + 1}/{len(songs)})")
            try:
                hit = self.find(artist, title)
            except ServiceError as exc:
                warnings_.append(str(exc))
                break  # stop hammering a failing service
            if hit is None:
                continue
            cached = self.cache.cache_get(self._cache_key(hit), ttl) if self.cache else None
            if cached is not None:
                c = json.loads(cached)
                results[i] = PreviewEstimate(
                    c["bpm"], c["camelot"], c["strength"], c["method"], hit
                )
            else:
                todo.append((i, hit))
        if not todo:
            return results, warnings_
        if on_progress:
            on_progress(f"estimating BPM and key from {len(todo)} Deezer previews")
        failed = 0
        with ThreadPoolExecutor(max_workers=dc.preview_workers) as pool:
            futures = [(i, pool.submit(self._estimate, hit)) for i, hit in todo]
            for i, future in futures:
                try:
                    estimate = future.result()
                except ServiceError:
                    failed += 1
                    continue
                results[i] = estimate
                if self.cache is not None:
                    body = {
                        "bpm": estimate.bpm,
                        "camelot": estimate.camelot,
                        "strength": estimate.key_strength,
                        "method": estimate.method,
                    }
                    self.cache.cache_put(self._cache_key(estimate.hit), json.dumps(body))
        if failed:
            warnings_.append(f"{failed} Deezer preview(s) could not be fetched or decoded")
        return results, warnings_


def _checked(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ServiceError("Deezer returned an unexpected response")
    if "error" in data:
        error = data["error"] if isinstance(data["error"], dict) else {}
        raise ServiceError(f"Deezer: {error.get('message', 'error')}")
    return data


def _fetch_preview(url: str, cfg: ScoringConfig) -> bytes:
    dc = cfg.discovery
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not (parts.hostname or "").endswith(dc.preview_host_suffix):
        raise ServiceError("preview URL is not on Deezer's CDN")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=dc.timeout_s) as response:
            data: bytes = response.read(dc.preview_max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ServiceError(f"could not fetch a Deezer preview: {exc}") from None
    if len(data) > dc.preview_max_bytes:
        raise ServiceError("Deezer preview is larger than expected")
    return data

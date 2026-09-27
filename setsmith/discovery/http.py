"""Small JSON-over-HTTPS client: polite pacing, SQLite response cache, no secrets in logs."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from setsmith import __version__

USER_AGENT = f"Setsmith/{__version__} (personal DJ tool; +https://github.com/)"
SECRET_PARAMS = frozenset({"api_key"})


class ServiceError(RuntimeError):
    """A discovery service refused or failed a request."""


class ResponseCache(Protocol):
    def cache_get(self, key: str, max_age_s: float) -> str | None: ...
    def cache_put(self, key: str, body: str) -> None: ...


class JsonClient:
    def __init__(
        self,
        name: str,
        base: str,
        *,
        min_interval_s: float,
        timeout_s: float,
        cache: ResponseCache | None = None,
        cache_ttl_s: float = 0.0,
    ) -> None:
        self.name = name
        self.base = base.rstrip("/") + "/"
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.cache = cache
        self.cache_ttl_s = cache_ttl_s
        self._last = 0.0
        self._lock = threading.Lock()
        self.requests_made = 0

    def _cache_key(self, path: str, params: dict[str, str]) -> str:
        public = sorted((k, v) for k, v in params.items() if k not in SECRET_PARAMS)
        return f"{self.name}:{path}?{urllib.parse.urlencode(public)}"

    def _pace(self) -> None:
        with self._lock:
            wait = self._last + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, path: str, params: dict[str, str]) -> Any:
        key = self._cache_key(path, params)
        if self.cache is not None:
            cached = self.cache.cache_get(key, self.cache_ttl_s)
            if cached is not None:
                return json.loads(cached)
        url = urllib.parse.urljoin(self.base, path.lstrip("/"))
        url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        self._pace()
        self.requests_made += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
            raise ServiceError(f"{self.name} answered HTTP {exc.code}: {_redact(detail)}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ServiceError(f"could not reach {self.name}: {reason}") from None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            raise ServiceError(f"{self.name} returned something that is not JSON") from None
        if self.cache is not None:
            self.cache.cache_put(key, body)
        return data


def _redact(text: str) -> str:
    return " ".join(w if "api_key" not in w else "api_key=***" for w in text.split())

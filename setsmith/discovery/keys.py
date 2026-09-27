"""API keys for discovery services: environment variables first, then a private file.

The file (~/.config/setsmith/keys.json, or under $SETSMITH_CONFIG_DIR) is created readable
by the owner only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SERVICES = {
    "lastfm": ("SETSMITH_LASTFM_KEY", "https://www.last.fm/api/account/create"),
    "getsongbpm": ("SETSMITH_GETSONGBPM_KEY", "https://getsongbpm.com/api"),
}


def config_dir() -> Path:
    if env := os.environ.get("SETSMITH_CONFIG_DIR"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "setsmith"


def keys_file() -> Path:
    return config_dir() / "keys.json"


def _read_file() -> dict[str, str]:
    try:
        data = json.loads(keys_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str)}


def get_key(service: str) -> str | None:
    env_name, _ = SERVICES[service]
    return os.environ.get(env_name) or _read_file().get(service) or None


def set_key(service: str, key: str) -> Path:
    if service not in SERVICES:
        raise ValueError(f"unknown service {service!r}; choose from {', '.join(SERVICES)}")
    path = keys_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_file()
    data[service] = key.strip()
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return path


def remove_key(service: str) -> bool:
    data = _read_file()
    if service not in data:
        return False
    del data[service]
    path = keys_file()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return True


def key_source(service: str) -> str | None:
    """'environment', 'file' or None: where a key would be read from."""
    env_name, _ = SERVICES[service]
    if os.environ.get(env_name):
        return "environment"
    return "file" if _read_file().get(service) else None

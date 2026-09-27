"""Read My Tags and play history from a *copy* of Rekordbox's master.db (optional).

Safety model:
- The live master.db is only ever read by a file copy. Setsmith copies it (plus any
  -wal/-shm files) into its own folder and opens that copy; the original is never opened.
- Nothing is written back: no commit is ever issued, and the copy is a throwaway.
- The database key comes from you (--key or $SETSMITH_REKORDBOX_KEY). Setsmith does not
  fetch keys and does not use any key bundled with pyrekordbox.

Needs the `rekordbox` extra (pyrekordbox).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from setsmith.analysis.audio import has_module

KEY_ENV = "SETSMITH_REKORDBOX_KEY"
_SIDECARS = ("-wal", "-shm")
_MYTAG_CATEGORY = 1  # DjmdMyTag.Attribute for a category (folder) row
_HISTORY_FOLDER = 1  # DjmdHistory.Attribute for a folder row


class RekordboxDbError(RuntimeError):
    """master.db could not be copied or read."""


def normalize_path(path: str | Path) -> str:
    """Comparable form of a file path (macOS may store accents decomposed)."""
    return unicodedata.normalize("NFC", str(path))


def default_master_db() -> Path | None:
    """Rekordbox 6/7's master.db in its default location, if it exists."""
    if sys.platform == "darwin":
        candidate = Path.home() / "Library" / "Pioneer" / "rekordbox" / "master.db"
    elif sys.platform == "win32":
        candidate = Path(os.environ.get("APPDATA", "")) / "Pioneer" / "rekordbox" / "master.db"
    else:
        return None
    return candidate if candidate.is_file() else None


def rekordbox_running() -> bool:
    if not has_module("psutil"):
        return False
    import psutil

    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if name.startswith("rekordbox"):
            return True
    return False


@dataclass(slots=True)
class HistorySession:
    id: str
    name: str
    date: str
    paths: list[str]  # normalized file paths, in play order


@dataclass(slots=True)
class RekordboxDbData:
    my_tags: dict[str, list[str]]  # normalized file path -> ["Category: Tag", ...]
    sessions: list[HistorySession]
    content_count: int
    copy_path: Path
    warnings: list[str] = field(default_factory=list)


def copy_master_db(source: Path, copies_dir: Path, keep: int) -> Path:
    """Copy master.db and its sidecar files into a new timestamped folder; prune old copies.

    Only folders inside `copies_dir` that Setsmith created are pruned.
    """
    if not source.is_file():
        raise RekordboxDbError(f"{source} not found")
    target_dir = copies_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target_dir.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source, target_dir / source.name)
        for suffix in _SIDECARS:
            sidecar = source.with_name(source.name + suffix)
            if sidecar.is_file():
                shutil.copy2(sidecar, target_dir / sidecar.name)
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise RekordboxDbError(f"could not copy {source}: {exc.strerror or exc}") from exc
    marker = target_dir / ".setsmith-copy"
    marker.write_text(str(source), encoding="utf-8")

    ours = sorted(d for d in copies_dir.iterdir() if (d / ".setsmith-copy").is_file())
    for old in ours[: max(0, len(ours) - keep)]:
        shutil.rmtree(old, ignore_errors=True)
    return target_dir / source.name


def read_master_db(copy_path: Path, key: str | None, *, unlock: bool = True) -> RekordboxDbData:
    """Read My Tags and history from a copied master.db. `unlock=False` is for test fixtures."""
    if not has_module("pyrekordbox"):
        raise RekordboxDbError("needs the 'rekordbox' extra: uv sync --extra rekordbox")
    if unlock and not key:
        raise RekordboxDbError(
            f"a database key is required: pass --key or set ${KEY_ENV} "
            "(Setsmith does not fetch keys or use bundled ones)"
        )
    from pyrekordbox import Rekordbox6Database
    from pyrekordbox.db6 import tables

    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)
    try:
        db = Rekordbox6Database(
            path=copy_path, db_dir=copy_path.parent, key=key or "", unlock=unlock
        )
    except Exception as exc:  # pyrekordbox raises ValueError, sqlite and sqlcipher errors
        raise RekordboxDbError(f"could not open the database copy: {exc}") from exc

    try:
        return _read(db, tables, copy_path)
    except Exception as exc:  # a wrong key surfaces here as "file is not a database"
        raise RekordboxDbError(f"could not read the database copy: {exc}") from exc
    finally:
        db.session.rollback()  # never commit anything, even to the copy
        db.close()


def _read(db: Any, tables: Any, copy_path: Path) -> RekordboxDbData:
    paths: dict[str, str] = {
        str(c.ID): normalize_path(c.FolderPath)
        for c in db.query(tables.DjmdContent.ID, tables.DjmdContent.FolderPath)
        if c.FolderPath
    }

    tag_rows = {str(t.ID): t for t in db.query(tables.DjmdMyTag)}

    def tag_label(tag_id: str) -> str | None:
        tag = tag_rows.get(tag_id)
        if tag is None or tag.Attribute == _MYTAG_CATEGORY:
            return None
        parent = tag_rows.get(str(tag.ParentID)) if tag.ParentID else None
        return f"{parent.Name}: {tag.Name}" if parent is not None else str(tag.Name)

    my_tags: dict[str, list[str]] = {}
    for row in db.query(tables.DjmdSongMyTag):
        path = paths.get(str(row.ContentID))
        label = tag_label(str(row.MyTagID))
        if path and label and label not in my_tags.setdefault(path, []):
            my_tags[path].append(label)

    sessions: list[HistorySession] = []
    for history in db.query(tables.DjmdHistory).order_by(tables.DjmdHistory.DateCreated):
        if history.Attribute == _HISTORY_FOLDER:
            continue
        songs = sorted(history.Songs, key=lambda s: s.TrackNo or 0)
        session_paths = [paths[str(s.ContentID)] for s in songs if str(s.ContentID) in paths]
        if session_paths:
            sessions.append(
                HistorySession(
                    str(history.ID),
                    str(history.Name or ""),
                    str(history.DateCreated or ""),
                    session_paths,
                )
            )
    return RekordboxDbData(my_tags, sessions, len(paths), copy_path)

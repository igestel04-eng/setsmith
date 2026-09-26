"""SQLite store: cached pair scores and listening feedback.

Default location is $SETSMITH_DB, else $XDG_DATA_HOME/setsmith/setsmith.db, else
~/.local/share/setsmith/setsmith.db. The store holds derived numbers and your own notes
only, never audio.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pair_scores (
    cfg TEXT NOT NULL,
    a_fp TEXT NOT NULL,
    b_fp TEXT NOT NULL,
    base REAL NOT NULL,
    boost INTEGER NOT NULL,
    PRIMARY KEY (cfg, a_fp, b_fp)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('good', 'bad')),
    a_id TEXT NOT NULL,
    b_id TEXT NOT NULL,
    a_label TEXT NOT NULL,
    b_label TEXT NOT NULL,
    a_location TEXT NOT NULL,
    b_location TEXT NOT NULL,
    transition_type TEXT NOT NULL,
    length_bars INTEGER,
    total REAL NOT NULL,
    score_json TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    collection TEXT NOT NULL DEFAULT ''
);
"""


def default_db_path() -> Path:
    if env := os.environ.get("SETSMITH_DB"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "setsmith" / "setsmith.db"


@dataclass(frozen=True, slots=True)
class FeedbackEntry:
    id: int
    created_at: str
    verdict: str
    a_id: str
    b_id: str
    a_label: str
    b_label: str
    transition_type: str
    length_bars: int | None
    total: float
    score: dict[str, Any]
    note: str
    collection: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "verdict": self.verdict,
            "from": {"id": self.a_id, "track": self.a_label},
            "to": {"id": self.b_id, "track": self.b_label},
            "transition_type": self.transition_type,
            "length_bars": self.length_bars,
            "total": self.total,
            "score": self.score,
            "note": self.note,
            "collection": self.collection,
        }


class Store:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ pair scores

    def load_pair_scores(
        self, cfg_key: str, fingerprints: Iterable[str]
    ) -> dict[tuple[str, str], tuple[float, bool]]:
        """All cached scores where both tracks are in `fingerprints`."""
        cur = self._conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS pool (fp TEXT PRIMARY KEY)")
        cur.execute("DELETE FROM pool")
        cur.executemany("INSERT OR IGNORE INTO pool VALUES (?)", ((fp,) for fp in fingerprints))
        rows = cur.execute(
            """SELECT s.a_fp, s.b_fp, s.base, s.boost FROM pair_scores s
               JOIN pool pa ON pa.fp = s.a_fp JOIN pool pb ON pb.fp = s.b_fp
               WHERE s.cfg = ?""",
            (cfg_key,),
        )
        return {(a, b): (base, bool(boost)) for a, b, base, boost in rows}

    def save_pair_scores(
        self, cfg_key: str, scores: dict[tuple[str, str], tuple[float, bool]]
    ) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO pair_scores VALUES (?, ?, ?, ?, ?)",
                ((cfg_key, a, b, base, int(boost)) for (a, b), (base, boost) in scores.items()),
            )

    def count_pair_scores(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM pair_scores").fetchone()
        return int(row[0])

    def clear_pair_scores(self) -> int:
        with self._conn:
            return self._conn.execute("DELETE FROM pair_scores").rowcount

    # ------------------------------------------------------------ feedback

    def add_feedback(
        self,
        *,
        verdict: str,
        a_id: str,
        b_id: str,
        a_label: str,
        b_label: str,
        a_location: str,
        b_location: str,
        transition_type: str,
        length_bars: int | None,
        total: float,
        score: dict[str, Any],
        note: str = "",
        collection: str = "",
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO feedback (created_at, verdict, a_id, b_id, a_label, b_label,
                   a_location, b_location, transition_type, length_bars, total, score_json,
                   note, collection) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    verdict,
                    a_id,
                    b_id,
                    a_label,
                    b_label,
                    a_location,
                    b_location,
                    transition_type,
                    length_bars,
                    total,
                    json.dumps(score),
                    note,
                    collection,
                ),
            )
        return int(cur.lastrowid or 0)

    def iter_feedback(self) -> Iterator[FeedbackEntry]:
        rows = self._conn.execute(
            """SELECT id, created_at, verdict, a_id, b_id, a_label, b_label, transition_type,
               length_bars, total, score_json, note, collection FROM feedback ORDER BY id"""
        )
        for (
            id_,
            created_at,
            verdict,
            a_id,
            b_id,
            a_label,
            b_label,
            transition_type,
            length_bars,
            total,
            score_json,
            note,
            collection,
        ) in rows:
            yield FeedbackEntry(
                id=id_,
                created_at=created_at,
                verdict=verdict,
                a_id=a_id,
                b_id=b_id,
                a_label=a_label,
                b_label=b_label,
                transition_type=transition_type,
                length_bars=length_bars,
                total=total,
                score=json.loads(score_json),
                note=note,
                collection=collection,
            )

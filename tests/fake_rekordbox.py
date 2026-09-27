"""An unencrypted stand-in for Rekordbox's master.db, built from pyrekordbox's own schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyrekordbox.db6 import tables
from sqlalchemy import create_engine, insert


def _row(table: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Fill NOT NULL columns the test does not care about with neutral values."""
    row = dict(values)
    for column in table.__table__.columns:
        if column.name in row or column.nullable or column.default is not None:
            continue
        try:
            numeric = column.type.python_type in (int, float)
        except NotImplementedError:
            numeric = False
        row[column.name] = 0 if numeric else ""
    return row


def make_master_db(
    path: Path,
    contents: dict[str, str],
    my_tags: dict[str, list[tuple[str, str]]],
    sessions: list[tuple[str, str, list[str]]],
) -> Path:
    """contents: ID -> FolderPath. my_tags: ID -> [(category, tag)].
    sessions: [(name, date, [content IDs in play order])]. Also adds one history folder.
    """
    engine = create_engine(f"sqlite:///{path}")
    tables.Base.metadata.create_all(engine)
    tag_ids: dict[tuple[str, str], str] = {}
    category_ids: dict[str, str] = {}
    with engine.begin() as conn:
        for cid, folder_path in contents.items():
            conn.execute(
                insert(tables.DjmdContent),
                [
                    _row(
                        tables.DjmdContent,
                        {"ID": cid, "FolderPath": folder_path, "Title": Path(folder_path).stem},
                    )
                ],
            )
        n = 0
        for cid, tags in my_tags.items():
            for category, tag in tags:
                if category not in category_ids:
                    category_ids[category] = f"cat{len(category_ids)}"
                    conn.execute(
                        insert(tables.DjmdMyTag),
                        [
                            _row(
                                tables.DjmdMyTag,
                                {"ID": category_ids[category], "Name": category, "Attribute": 1},
                            )
                        ],
                    )
                if (category, tag) not in tag_ids:
                    tag_ids[(category, tag)] = f"tag{len(tag_ids)}"
                    conn.execute(
                        insert(tables.DjmdMyTag),
                        [
                            _row(
                                tables.DjmdMyTag,
                                {
                                    "ID": tag_ids[(category, tag)],
                                    "Name": tag,
                                    "Attribute": 0,
                                    "ParentID": category_ids[category],
                                },
                            )
                        ],
                    )
                n += 1
                conn.execute(
                    insert(tables.DjmdSongMyTag),
                    [
                        _row(
                            tables.DjmdSongMyTag,
                            {
                                "ID": f"smt{n}",
                                "MyTagID": tag_ids[(category, tag)],
                                "ContentID": cid,
                                "TrackNo": n,
                            },
                        )
                    ],
                )
        conn.execute(
            insert(tables.DjmdHistory),
            [
                _row(
                    tables.DjmdHistory,
                    {"ID": "folder", "Name": "2026", "Attribute": 1, "DateCreated": "2026-01-01"},
                )
            ],
        )
        for i, (name, date, ids) in enumerate(sessions):
            conn.execute(
                insert(tables.DjmdHistory),
                [
                    _row(
                        tables.DjmdHistory,
                        {
                            "ID": f"h{i}",
                            "Name": name,
                            "Attribute": 0,
                            "DateCreated": date,
                            "ParentID": "folder",
                        },
                    )
                ],
            )
            # Insert out of order to prove play order comes from TrackNo.
            for pos, cid in reversed(list(enumerate(ids, 1))):
                conn.execute(
                    insert(tables.DjmdSongHistory),
                    [
                        _row(
                            tables.DjmdSongHistory,
                            {
                                "ID": f"sh{i}-{pos}",
                                "HistoryID": f"h{i}",
                                "ContentID": cid,
                                "TrackNo": pos,
                            },
                        )
                    ],
                )
    engine.dispose()
    return path

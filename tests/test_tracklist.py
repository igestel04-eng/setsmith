from pathlib import Path

import pytest

from setsmith.analysis.tracklist import (
    match_tracklist,
    parse_time,
    parse_tracklist,
    timestamps_usable,
)
from setsmith.model.collection import Collection, normalize_text, strip_versions

TEXT = """Tracklist:
01. Kaya Sol - Sunwater (Extended Mix) [Tidewater]
[00:06:10] Moana Reyes \u2013 Lanterns
3) 0:12:40 Nils Ødegaard - Glass Harbour (Café Remix)
w/ Dex Marlo - Floor Tom
1:02:33 ID - ID
-----
Just some words
"""


def test_parse_text_formats() -> None:
    entries = parse_tracklist(TEXT)
    assert [(e.position, e.artist, e.title) for e in entries] == [
        (1, "Kaya Sol", "Sunwater (Extended Mix)"),
        (2, "Moana Reyes", "Lanterns"),
        (3, "Nils Ødegaard", "Glass Harbour (Café Remix)"),
        (4, "Dex Marlo", "Floor Tom"),
        (5, "ID", "ID"),
    ]
    first, second, third, fourth, fifth = entries
    assert first.label == "Tidewater" and first.start_s is None
    assert second.start_s == 370 and third.start_s == 760
    assert fourth.together and not third.together
    assert fifth.unknown and fifth.start_s == 3753


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("0:00", 0.0), ("6:10", 370.0), ("1:02:33", 3753.0), ("75:10", 4510.0), ("6:61", None)],
)
def test_parse_time(text: str, seconds: float | None) -> None:
    assert parse_time(text) == seconds


def test_parse_csv() -> None:
    csv_text = "Artist;Title;Start\nKaya Sol;Sunwater;0:00\nMoana Reyes;Lanterns;6:10\nID;ID;9:00\n"
    entries = parse_tracklist(csv_text)
    assert [(e.artist, e.title, e.start_s, e.unknown) for e in entries] == [
        ("Kaya Sol", "Sunwater", 0.0, False),
        ("Moana Reyes", "Lanterns", 370.0, False),
        ("ID", "ID", 540.0, True),
    ]
    assert timestamps_usable(entries)


def test_timestamps_usable() -> None:
    assert not timestamps_usable(parse_tracklist("0:00 A - B\nC - D\n"))
    assert not timestamps_usable(parse_tracklist("5:00 A - B\n1:00 C - D\n"))
    assert timestamps_usable(parse_tracklist("0:00 A - B\n5:00 C - D\n"))


def test_strip_versions() -> None:
    assert strip_versions(normalize_text("Gamma (Original Mix)")) == "gamma"
    assert strip_versions(normalize_text("Gamma (Extended)")) == "gamma"
    assert strip_versions(normalize_text("Gamma (Dixon Remix)")) == "gamma dixon remix"
    assert strip_versions("original") == "original"  # never strips a title to nothing


def test_match_tracklist(collection_5: Collection) -> None:
    text = (
        "Kaya Sol - Sunwater (Original Mix)\n"  # collection has "(Extended Mix)"
        "Moana Reyes - Lanterns\n"
        "ID - ID\n"
        "Nobody - Nothing Here\n"
    )
    matches = match_tracklist(collection_5, parse_tracklist(text))
    assert [m.track.id if m.track else None for m in matches] == ["101", "102", None, None]
    assert matches[2].entry.unknown and matches[2].score == 0.0


def test_matching_duplicates_is_flagged(collection_50: Collection) -> None:
    (m,) = match_tracklist(collection_50, parse_tracklist("Kaya Sol - Tidal Drum\n"))
    assert m.track is not None and m.ambiguous  # TrackIDs 1 and 5 are the same song


def test_load_tracklist_bom(tmp_path: Path) -> None:
    from setsmith.analysis.tracklist import load_tracklist

    path = tmp_path / "t.txt"
    path.write_bytes("﻿01. A - B\n".encode())
    assert load_tracklist(path)[0].artist == "A"

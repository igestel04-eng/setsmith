from pathlib import Path
from typing import Any

import pytest
from lxml import etree

from setsmith.io.rekordbox_xml import (
    SETSMITH_FOLDER,
    ExportError,
    PlaylistSpec,
    export_playlists,
    load_collection,
    read_product_name,
)
from setsmith.model.collection import KeyType, NodeType


def raw_tracks(path: Path) -> dict[str, Any]:
    """TrackID -> (attributes, [(child tag, child attributes)]) straight from the XML."""
    root = etree.parse(str(path)).getroot()
    return {
        t.get("TrackID"): (dict(t.attrib), [(c.tag, dict(c.attrib)) for c in t])
        for t in root.find("COLLECTION")
    }


def setsmith_playlists(path: Path) -> dict[str, list[str]]:
    root = load_collection(path).playlists
    assert root is not None
    (folder,) = [n for n in root.children if n.name == SETSMITH_FOLDER]
    assert folder.type == NodeType.FOLDER
    out = {}
    for node in folder.children:
        assert node.type == NodeType.PLAYLIST and node.key_type == KeyType.TRACK_ID
        out[node.name] = node.track_keys
    return out


def test_round_trip_preserves_tracks_exactly(fixture_5: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    result = export_playlists(fixture_5, out, [PlaylistSpec("Test", ("104", "101", "103"))])
    assert result.track_count == 3 and result.written == ["Test"]

    source, exported = raw_tracks(fixture_5), raw_tracks(out)
    assert set(exported) == {"101", "103", "104"}
    for tid, data in exported.items():
        assert data == source[tid]  # every attribute, TEMPO and POSITION_MARK unchanged

    original, reloaded = load_collection(fixture_5), load_collection(out)
    for tid in exported:
        assert reloaded.tracks[tid] == original.tracks[tid]
    assert setsmith_playlists(out) == {"Test": ["104", "101", "103"]}
    assert read_product_name(out) == "Setsmith"


def test_entries_counts(fixture_5: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    export_playlists(
        fixture_5, out, [PlaylistSpec("A", ("101", "102")), PlaylistSpec("B", ("102", "103"))]
    )
    root = etree.parse(str(out)).getroot()
    assert root.find("COLLECTION").get("Entries") == "3"
    folder = root.find("PLAYLISTS/NODE/NODE")
    assert folder.get("Name") == "Setsmith" and folder.get("Count") == "2"
    for node in folder:
        assert node.get("Entries") == str(len(node))
        assert node.get("Type") == "1" and node.get("KeyType") == "0"


def test_never_overwrites_input(fixture_5: Path) -> None:
    before = fixture_5.read_bytes()
    with pytest.raises(ExportError, match="input"):
        export_playlists(fixture_5, fixture_5, [PlaylistSpec("X", ("101",))])
    alias = fixture_5.parent / "." / fixture_5.name
    with pytest.raises(ExportError, match="input"):
        export_playlists(fixture_5, alias, [PlaylistSpec("X", ("101",))])
    assert fixture_5.read_bytes() == before


def test_refuses_foreign_file_without_force(fixture_5: Path, tmp_path: Path) -> None:
    foreign = tmp_path / "mine.xml"
    foreign.write_bytes(fixture_5.read_bytes())  # a real Rekordbox export
    with pytest.raises(ExportError, match="not written by Setsmith"):
        export_playlists(fixture_5, foreign, [PlaylistSpec("X", ("101",))])
    export_playlists(fixture_5, foreign, [PlaylistSpec("X", ("101",))], overwrite=True)
    assert read_product_name(foreign) == "Setsmith"


def test_merges_into_existing_setsmith_file(fixture_5: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    export_playlists(fixture_5, out, [PlaylistSpec("Friday", ("101", "102"))])
    result = export_playlists(fixture_5, out, [PlaylistSpec("Saturday", ("103", "104"))])
    assert result.kept == ["Friday"]
    assert setsmith_playlists(out) == {"Friday": ["101", "102"], "Saturday": ["103", "104"]}
    export_playlists(fixture_5, out, [PlaylistSpec("Friday", ("104",))])  # same name: replaced
    assert setsmith_playlists(out) == {"Saturday": ["103", "104"], "Friday": ["104"]}
    assert set(raw_tracks(out)) == {"103", "104"}  # no orphaned tracks


def test_comments_only_change_in_output(fixture_5: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    export_playlists(
        fixture_5, out, [PlaylistSpec("C", ("101", "102"))], comments={"101": "[Setsmith #1]"}
    )
    exported = raw_tracks(out)
    assert exported["101"][0]["Comments"] == "[Setsmith #1]"
    assert exported["102"][0]["Comments"] == "8A - Energy 7"
    assert raw_tracks(fixture_5)["101"][0]["Comments"] == "E6 shakers, warm vocal chant"


def test_errors(fixture_5: Path, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="999"):
        export_playlists(fixture_5, tmp_path / "a.xml", [PlaylistSpec("X", ("101", "999"))])
    with pytest.raises(ExportError, match="unique"):
        export_playlists(
            fixture_5, tmp_path / "b.xml", [PlaylistSpec("X", ("101",)), PlaylistSpec("X", ())]
        )
    assert not (tmp_path / "a.xml").exists()


def test_large_collection_export(fixture_pool: Path, tmp_path: Path) -> None:
    out = tmp_path / "set.xml"
    ids = tuple(str(i) for i in range(1, 300, 7))
    export_playlists(fixture_pool, out, [PlaylistSpec("Big", ids)])
    source = raw_tracks(fixture_pool)
    assert raw_tracks(out) == {tid: source[tid] for tid in ids}

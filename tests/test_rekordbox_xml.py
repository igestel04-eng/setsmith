from pathlib import Path

import pytest

from setsmith.io.rekordbox_xml import RekordboxXMLError, load_collection, rating_to_stars
from setsmith.model.collection import Collection, KeyType, NodeType
from setsmith.model.track import CueType, EnergySource, TempoMarker, derive_energy


def test_loads_all_tracks(collection_5: Collection) -> None:
    assert len(collection_5) == 5
    assert list(collection_5.tracks) == ["101", "102", "103", "104", "105"]
    assert collection_5.product == "rekordbox 7.0.4"


def test_track_attributes(collection_5: Collection) -> None:
    t = collection_5.tracks["101"]
    assert t.title == "Sunwater (Extended Mix)"
    assert t.artist == "Kaya Sol"
    assert t.genre == "Afro House"
    assert t.label == "Tidewater"
    assert t.duration_s == 412
    assert t.bpm == 122.0
    assert t.key_raw == "Am"
    assert t.camelot == "8A"
    assert t.key_confidence == 1.0
    assert t.rating == 4
    assert t.colour == "0xFF007F"
    assert t.location == (
        "file://localhost/Users/dj/Music/Afro/Kaya%20Sol%20-%20Sunwater%20(Extended%20Mix).mp3"
    )


def test_unicode_and_escaped_entities(collection_5: Collection) -> None:
    t = collection_5.tracks["103"]
    assert t.artist == "Nils Ødegaard"
    assert t.remixer == "Café"
    assert t.genre == "Melodic House & Techno"
    assert "%C3%98" in t.location  # URI kept exactly as read, not decoded


def test_tonality_notations(collection_5: Collection) -> None:
    assert collection_5.tracks["102"].camelot == "9A"  # already Camelot
    assert collection_5.tracks["103"].camelot == "8B"  # "C"
    assert collection_5.tracks["104"].camelot == "11A"  # "F#m"


def test_missing_data_is_none_not_error(collection_5: Collection) -> None:
    t = collection_5.tracks["105"]
    assert t.bpm is None  # AverageBpm="0.00"
    assert t.key_raw is None and t.camelot is None  # Tonality=""
    assert t.energy is None
    assert t.rating == 0
    assert t.artist == ""
    assert t.display == "untitled_final_v3"


def test_beat_grid(collection_5: Collection) -> None:
    t = collection_5.tracks["101"]
    assert t.beat_grid == [TempoMarker(inizio_s=0.071, bpm=122.0, metro="4/4", battito=1)]
    assert not t.variable_tempo
    varying = collection_5.tracks["104"]
    assert [m.bpm for m in varying.beat_grid] == [127.5, 128.0]
    assert varying.variable_tempo


def test_cue_points(collection_5: Collection) -> None:
    cues = collection_5.tracks["101"].cues
    assert len(cues) == 3
    memory, hot, loop = cues
    assert memory.is_memory_cue and memory.type == CueType.CUE and memory.start_s == 0.071
    assert not hot.is_memory_cue and hot.num == 0 and hot.name == "Drop"
    assert loop.type == CueType.LOOP and loop.start_s == 362.301 and loop.end_s == 370.169
    assert memory.end_s is None


def test_energy_sources(collection_5: Collection) -> None:
    assert collection_5.tracks["101"].energy == 6  # "E6 shakers..."
    assert collection_5.tracks["101"].energy_source == EnergySource.COMMENT_TAG
    assert collection_5.tracks["102"].energy == 7  # Mixed In Key "8A - Energy 7"
    assert collection_5.tracks["103"].energy == 6  # 3 stars x 2
    assert collection_5.tracks["103"].energy_source == EnergySource.RATING
    assert collection_5.tracks["104"].energy is None  # unrated, no tag


@pytest.mark.parametrize(
    ("comments", "rating", "expected"),
    [
        ("E10 peak", 0, 10),
        ("e3", 5, 3),  # tag wins over rating
        ("Energy 8", 0, 8),
        ("E11", 0, None),  # out of scale
        ("RE7 remix", 0, None),  # not a standalone tag
        ("", 5, 10),
        ("", 1, 2),
        ("", 0, None),
    ],
)
def test_derive_energy(comments: str, rating: int, expected: float | None) -> None:
    assert derive_energy(comments, rating)[0] == expected


@pytest.mark.parametrize(
    ("raw", "stars"),
    [("0", 0), ("51", 1), ("102", 2), ("153", 3), ("204", 4), ("255", 5), ("", 0), ("300", 5)],
)
def test_rating_mapping(raw: str, stars: int) -> None:
    assert rating_to_stars(raw) == stars


def test_playlist_tree(collection_5: Collection) -> None:
    root = collection_5.playlists
    assert root is not None
    assert root.name == "ROOT" and root.type == NodeType.FOLDER
    paths = {"/".join(p): node for p, node in root.walk()}
    assert set(paths) == {"ROOT", "ROOT/Warm Up", "ROOT/Gigs", "ROOT/Gigs/Friday"}

    warm_up = paths["ROOT/Warm Up"]
    assert warm_up.type == NodeType.PLAYLIST and warm_up.key_type == KeyType.TRACK_ID
    assert [t.id for t in collection_5.playlist_tracks(warm_up)] == ["101", "102"]

    friday = paths["ROOT/Gigs/Friday"]
    assert friday.key_type == KeyType.LOCATION
    assert [t.id for t in collection_5.playlist_tracks(friday)] == ["103"]


def test_warnings_for_bad_data(tmp_path: Path) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0"><COLLECTION Entries="4">
  <TRACK TrackID="1" Name="A" Tonality="Xyz" AverageBpm="12o"/>
  <TRACK TrackID="1" Name="A duplicate"/>
  <TRACK Name="No id"/>
  <TRACK TrackID="2" Name="B" Tonality="Xyz" Rating="garbage">
    <TEMPO Inizio="" Bpm="120"/><POSITION_MARK Type="9" Start="1.0"/>
  </TRACK>
</COLLECTION></DJ_PLAYLISTS>""",
        encoding="utf-8",
    )
    c = load_collection(xml)
    assert set(c.tracks) == {"1", "2"}
    assert c.tracks["1"].title == "A"
    assert c.tracks["1"].bpm is None
    assert c.tracks["2"].beat_grid == []  # marker without Inizio dropped
    assert c.tracks["2"].cues[0].type == CueType.CUE  # unknown Type falls back
    assert c.playlists is None
    joined = "\n".join(c.warnings)
    assert "Duplicate TrackID 1" in joined
    assert "without TrackID" in joined
    assert "'Xyz' on 2 track(s)" in joined


def test_rejects_non_rekordbox_xml(tmp_path: Path) -> None:
    other = tmp_path / "other.xml"
    other.write_text("<plist><dict/></plist>", encoding="utf-8")
    with pytest.raises(RekordboxXMLError, match="expected <DJ_PLAYLISTS>"):
        load_collection(other)

    broken = tmp_path / "broken.xml"
    broken.write_text("<DJ_PLAYLISTS><COLLECTION>", encoding="utf-8")
    with pytest.raises(RekordboxXMLError, match="not valid XML"):
        load_collection(broken)

    with pytest.raises(FileNotFoundError):
        load_collection(tmp_path / "missing.xml")


def test_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    xml = tmp_path / "xxe.xml"
    xml.write_text(
        f"""<?xml version="1.0"?>
<!DOCTYPE DJ_PLAYLISTS [<!ENTITY x SYSTEM "file://{secret}">]>
<DJ_PLAYLISTS><COLLECTION><TRACK TrackID="1" Name="&x;"/></COLLECTION></DJ_PLAYLISTS>""",
        encoding="utf-8",
    )
    with pytest.raises(RekordboxXMLError, match="external entity"):
        load_collection(xml)


def test_find_track(collection_5: Collection) -> None:
    assert collection_5.find("Kaya Sol - Sunwater").track is not None
    assert collection_5.find("kaya sol - sunwater (extended mix)").track.id == "101"  # type: ignore[union-attr]
    assert collection_5.find("Nils Odegaard - Glass Harbour").track.id == "103"  # type: ignore[union-attr]
    assert collection_5.find("Lanterns").track.id == "102"  # type: ignore[union-attr]
    assert collection_5.find("104").track.id == "104"  # type: ignore[union-attr]
    missing = collection_5.find("Nobody - Nothing Here")
    assert missing.track is None and not missing.ambiguous

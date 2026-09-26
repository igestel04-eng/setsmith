"""Read Rekordbox XML exports and write Setsmith playlists as new Rekordbox XML files.

Import is streaming (lxml iterparse) so 50,000+ track collections load in bounded memory.
Export copies each referenced TRACK element byte-for-byte in meaning (every attribute,
TEMPO and POSITION_MARK unchanged, Location exactly as read) into a new file. The input
file is never written.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from setsmith import __version__
from setsmith.keys.camelot import normalize_key
from setsmith.model.collection import Collection, KeyType, NodeType, PlaylistNode
from setsmith.model.track import CuePoint, CueType, TempoMarker, Track, derive_energy
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig

# Rekordbox stores star ratings as 0, 51, 102, 153, 204, 255.
RATING_STEP = 51
MAX_STARS = 5


class RekordboxXMLError(ValueError):
    """The file is not a readable Rekordbox XML export."""


def _attr(elem: Any, name: str) -> str:
    value = elem.get(name)
    return value if isinstance(value, str) else ""


def _float(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(text: str, default: int = 0) -> int:
    value = _float(text)
    return int(value) if value is not None else default


def rating_to_stars(raw: str) -> int:
    value = _int(raw)
    return max(0, min(MAX_STARS, round(value / RATING_STEP)))


def _parse_tempo(elem: Any) -> TempoMarker | None:
    bpm = _float(_attr(elem, "Bpm"))
    inizio = _float(_attr(elem, "Inizio"))
    if bpm is None or inizio is None:
        return None
    return TempoMarker(
        inizio_s=inizio,
        bpm=bpm,
        metro=_attr(elem, "Metro") or "4/4",
        battito=_int(_attr(elem, "Battito"), default=1),
    )


def _parse_cue(elem: Any) -> CuePoint | None:
    start = _float(_attr(elem, "Start"))
    if start is None:
        return None
    type_value = _int(_attr(elem, "Type"))
    try:
        cue_type = CueType(type_value)
    except ValueError:
        cue_type = CueType.CUE
    return CuePoint(
        name=_attr(elem, "Name"),
        type=cue_type,
        start_s=start,
        end_s=_float(_attr(elem, "End")),
        num=_int(_attr(elem, "Num"), default=-1),
    )


def parse_track(elem: Any, cfg: ScoringConfig = DEFAULT_CONFIG) -> Track:
    """Build a Track from a COLLECTION > TRACK element. Missing data becomes None."""
    bpm = _float(_attr(elem, "AverageBpm"))
    key_raw = _attr(elem, "Tonality").strip() or None
    rating = rating_to_stars(_attr(elem, "Rating"))
    comments = _attr(elem, "Comments")
    energy, energy_source = derive_energy(comments, rating, cfg.energy)

    beat_grid = [m for t in elem.iterfind("TEMPO") if (m := _parse_tempo(t)) is not None]
    cues = [c for p in elem.iterfind("POSITION_MARK") if (c := _parse_cue(p)) is not None]
    grid_bpms = sorted({m.bpm for m in beat_grid})
    variable_tempo = (
        len(grid_bpms) > 1 and grid_bpms[-1] - grid_bpms[0] >= cfg.tempo.variable_tempo_min_bpm_diff
    )

    return Track(
        id=_attr(elem, "TrackID"),
        title=_attr(elem, "Name"),
        artist=_attr(elem, "Artist"),
        remixer=_attr(elem, "Remixer"),
        label=_attr(elem, "Label"),
        genre=_attr(elem, "Genre"),
        comments=comments,
        location=_attr(elem, "Location"),
        duration_s=_float(_attr(elem, "TotalTime")) or 0.0,
        bpm=bpm if bpm else None,  # Rekordbox writes 0.00 for "not analyzed"
        key_raw=key_raw,
        camelot=normalize_key(key_raw),
        rating=rating,
        colour=_attr(elem, "Colour") or None,
        beat_grid=beat_grid,
        cues=cues,
        energy=energy,
        energy_source=energy_source,
        variable_tempo=variable_tempo,
    )


def _parse_node(elem: Any) -> PlaylistNode:
    try:
        node_type = NodeType(_int(_attr(elem, "Type")))
    except ValueError:
        node_type = NodeType.FOLDER
    try:
        key_type = KeyType(_int(_attr(elem, "KeyType")))
    except ValueError:
        key_type = KeyType.TRACK_ID
    node = PlaylistNode(name=_attr(elem, "Name"), type=node_type, key_type=key_type)
    if node_type == NodeType.FOLDER:
        node.children = [_parse_node(child) for child in elem.iterfind("NODE")]
    else:
        node.track_keys = [key for t in elem.iterfind("TRACK") if (key := _attr(t, "Key"))]
    return node


def _release(elem: Any) -> None:
    """Free an element and its already-processed preceding siblings."""
    elem.clear()
    parent = elem.getparent()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]


def load_collection(path: str | Path, cfg: ScoringConfig = DEFAULT_CONFIG) -> Collection:
    """Stream-parse a Rekordbox XML export."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    tracks: dict[str, Track] = {}
    warnings: list[str] = []
    playlists: PlaylistNode | None = None
    product = ""
    unparsed_keys: dict[str, int] = {}
    root_seen = False
    in_collection = False

    context = etree.iterparse(
        str(path),
        events=("start", "end"),
        resolve_entities=False,
        no_network=True,
        remove_comments=True,
    )
    try:
        for event, elem in context:
            tag = elem.tag
            if event == "start":
                if not root_seen:
                    root_seen = True
                    if tag != "DJ_PLAYLISTS":
                        raise RekordboxXMLError(
                            f"{path.name}: root element is <{tag}>, expected <DJ_PLAYLISTS>"
                        )
                elif tag == "COLLECTION":
                    in_collection = True
                continue

            if tag == "TRACK" and in_collection:
                track = parse_track(elem, cfg)
                if not track.id:
                    warnings.append(f"Skipped a TRACK without TrackID ({track.display})")
                elif track.id in tracks:
                    warnings.append(f"Duplicate TrackID {track.id}: kept the first entry")
                else:
                    tracks[track.id] = track
                    if track.key_raw and track.camelot is None:
                        unparsed_keys[track.key_raw] = unparsed_keys.get(track.key_raw, 0) + 1
                _release(elem)
            elif tag == "COLLECTION":
                in_collection = False
                _release(elem)
            elif tag == "PRODUCT":
                product = " ".join(v for v in (_attr(elem, "Name"), _attr(elem, "Version")) if v)
            elif tag == "PLAYLISTS":
                root_node = elem.find("NODE")
                if root_node is not None:
                    playlists = _parse_node(root_node)
                _release(elem)
    except etree.XMLSyntaxError as exc:
        raise RekordboxXMLError(f"{path.name}: not valid XML ({exc})") from exc

    if not root_seen:
        raise RekordboxXMLError(f"{path.name}: empty file")
    for raw, count in sorted(unparsed_keys.items(), key=lambda kv: -kv[1]):
        warnings.append(f"Unrecognized Tonality {raw!r} on {count} track(s): treated as missing")

    return Collection(
        tracks=tracks, playlists=playlists, source_path=path, product=product, warnings=warnings
    )


# ---------------------------------------------------------------- export

SETSMITH_PRODUCT = "Setsmith"
SETSMITH_FOLDER = "Setsmith"


class ExportError(ValueError):
    """The export would be unsafe (overwrite the input or a non-Setsmith file) or incomplete."""


@dataclass(frozen=True, slots=True)
class PlaylistSpec:
    name: str
    track_ids: tuple[str, ...]


@dataclass(slots=True)
class ExportResult:
    path: Path
    written: list[str]  # playlist names written by this export
    kept: list[str] = field(default_factory=list)  # earlier Setsmith playlists kept in the file
    track_count: int = 0


def _iter_collection_tracks(path: Path) -> Any:
    """Yield (TrackID, TRACK element) for each COLLECTION track. Elements are freed after use."""
    in_collection = False
    context = etree.iterparse(
        str(path), events=("start", "end"), resolve_entities=False, no_network=True
    )
    for event, elem in context:
        if event == "start":
            if elem.tag == "COLLECTION":
                in_collection = True
            continue
        if elem.tag == "TRACK" and in_collection:
            yield _attr(elem, "TrackID"), elem
            _release(elem)
        elif elem.tag == "COLLECTION":
            return


def _copy_tracks(path: Path, wanted: set[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not wanted:
        return found
    for track_id, elem in _iter_collection_tracks(path):
        if track_id in wanted and track_id not in found:
            found[track_id] = copy.deepcopy(elem)
            if len(found) == len(wanted):
                break
    return found


def read_product_name(path: Path) -> str:
    """PRODUCT Name of a Rekordbox-style XML file, or "" if absent or unreadable."""
    try:
        for _, elem in etree.iterparse(str(path), events=("end",), resolve_entities=False):
            if elem.tag == "PRODUCT":
                return _attr(elem, "Name")
            if elem.tag in ("COLLECTION", "PLAYLISTS"):
                break
    except etree.XMLSyntaxError:
        pass
    return ""


def _existing_setsmith_playlists(path: Path) -> list[PlaylistSpec]:
    playlists = load_collection(path).playlists
    if playlists is None:
        return []
    out: list[PlaylistSpec] = []
    for folder in playlists.children:
        if folder.type == NodeType.FOLDER and folder.name == SETSMITH_FOLDER:
            out.extend(
                PlaylistSpec(node.name, tuple(node.track_keys))
                for node in folder.children
                if node.type == NodeType.PLAYLIST and node.key_type == KeyType.TRACK_ID
            )
    return out


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    return a.exists() and b.exists() and os.path.samefile(a, b)


def export_playlists(
    source: str | Path,
    out: str | Path,
    playlists: list[PlaylistSpec],
    *,
    comments: dict[str, str] | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Write `playlists` into a Setsmith folder in a new Rekordbox XML file.

    If `out` already holds Setsmith output, its other Setsmith playlists are kept and a
    playlist with the same name is replaced. Any other existing file is refused unless
    `overwrite`. The source file is never written. `comments` replaces the Comments
    attribute of the given TrackIDs in the output only.
    """
    source, out = Path(source), Path(out)
    if _same_file(source, out):
        raise ExportError(f"refusing to overwrite the input file {source}")
    names = [p.name for p in playlists]
    if len(set(names)) != len(names):
        raise ExportError("playlist names must be unique")

    kept: list[PlaylistSpec] = []
    if out.exists():
        if read_product_name(out) == SETSMITH_PRODUCT:
            kept = [p for p in _existing_setsmith_playlists(out) if p.name not in names]
        elif not overwrite:
            raise ExportError(f"{out} exists and was not written by Setsmith; use --force")

    new_ids = {tid for p in playlists for tid in p.track_ids}
    elements = _copy_tracks(source, new_ids)
    missing = new_ids - elements.keys()
    if missing:
        raise ExportError(f"TrackIDs not found in {source.name}: {', '.join(sorted(missing))}")
    kept_ids = {tid for p in kept for tid in p.track_ids} - elements.keys()
    elements.update(_copy_tracks(out, kept_ids))
    kept = [p for p in kept if all(tid in elements for tid in p.track_ids)]

    all_lists = [*kept, *playlists]
    order: list[str] = []
    for p in all_lists:
        order.extend(tid for tid in p.track_ids if tid not in order)

    root = etree.Element("DJ_PLAYLISTS", Version="1.0.0")
    etree.SubElement(root, "PRODUCT", Name=SETSMITH_PRODUCT, Version=__version__, Company="")
    collection = etree.SubElement(root, "COLLECTION", Entries=str(len(order)))
    for tid in order:
        elem = elements[tid]
        if comments and tid in comments:
            elem.set("Comments", comments[tid])
        collection.append(elem)

    tree = etree.SubElement(root, "PLAYLISTS")
    root_node = etree.SubElement(tree, "NODE", Type="0", Name="ROOT", Count="1")
    folder = etree.SubElement(
        root_node, "NODE", Type="0", Name=SETSMITH_FOLDER, Count=str(len(all_lists))
    )
    for p in all_lists:
        node = etree.SubElement(
            folder, "NODE", Name=p.name, Type="1", KeyType="0", Entries=str(len(p.track_ids))
        )
        for tid in p.track_ids:
            etree.SubElement(node, "TRACK", Key=tid)

    etree.indent(root, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    etree.ElementTree(root).write(str(tmp), encoding="UTF-8", xml_declaration=True)
    os.replace(tmp, out)
    return ExportResult(out, names, [p.name for p in kept], len(order))

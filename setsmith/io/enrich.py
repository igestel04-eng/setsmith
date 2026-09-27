"""Apply imported Rekordbox My Tags to a loaded collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from setsmith.analysis.audio import location_to_path
from setsmith.io.rekordbox_db import normalize_path
from setsmith.model.collection import Collection
from setsmith.model.track import Track, derive_vocal
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig


@dataclass(slots=True)
class MyTagCoverage:
    tagged: int = 0
    vocal_from_tags: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def tag_names(track: Track) -> set[str]:
    """Lowercased tag names, with and without their category ("vibe: dark" and "dark")."""
    names = set()
    for tag in track.my_tags:
        names.add(tag.casefold())
        names.add(tag.split(": ", 1)[-1].casefold())
    return names


def apply_my_tags(
    collection: Collection, my_tags: dict[str, list[str]], cfg: ScoringConfig = DEFAULT_CONFIG
) -> MyTagCoverage:
    """Attach My Tags by file path. A vocal/instrumental My Tag overrides title/comment words."""
    cov = MyTagCoverage()
    if not my_tags:
        return cov
    for track in collection.tracks.values():
        path = location_to_path(track.location)
        tags = my_tags.get(normalize_path(path)) if path else None
        if not tags:
            continue
        track.my_tags = list(tags)
        cov.tagged += 1
        vocal = derive_vocal(*(t.split(": ", 1)[-1] for t in tags), cfg=cfg.vocal)
        if vocal is not None:
            track.vocal = vocal
            cov.vocal_from_tags += 1
    return cov

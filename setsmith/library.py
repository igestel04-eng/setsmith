"""Shared helpers for front ends (CLI, web): enrich a loaded collection with everything
Setsmith has stored about it, and build the scoring config to use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from setsmith.analysis.apply import AnalysisCoverage, load_and_apply
from setsmith.analysis.learn import LearnedPreferences, blended_config
from setsmith.io.enrich import MyTagCoverage, apply_my_tags
from setsmith.model.collection import Collection
from setsmith.scoring.weights import ScoringConfig
from setsmith.store import Store, default_db_path


@dataclass(slots=True)
class Enrichment:
    analysis: AnalysisCoverage | None = None
    my_tags: MyTagCoverage | None = None
    rekordbox: dict[str, Any] | None = None  # last master.db import


def enrich(collection: Collection, db: Path | None, use_analysis: bool = True) -> Enrichment:
    """Apply stored audio analysis and imported My Tags, if any, in place.

    Never creates a database just to read from it.
    """
    info = Enrichment()
    db_path = db or default_db_path()
    if not db_path.exists():
        return info
    with Store(db_path) as store:
        if use_analysis:
            info.analysis = load_and_apply(collection, store)
        info.rekordbox = store.rekordbox_import_info()
        if info.rekordbox:
            info.my_tags = apply_my_tags(collection, store.load_my_tags())
    return info


def learned_config(db: Path | None) -> ScoringConfig | None:
    """Scoring config blended with learned preferences, or None if nothing is learned."""
    db_path = db or default_db_path()
    if not db_path.exists():
        return None
    with Store(db_path) as store:
        data = store.load_learned()
    return blended_config(LearnedPreferences.from_dict(data)) if data else None

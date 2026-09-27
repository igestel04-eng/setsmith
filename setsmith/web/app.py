"""Local web UI: build sets on a timeline, get suggestions, browse analyzed live sets.

Runs on 127.0.0.1 by default. It reads the collection given at startup (never writes it),
only accepts requests addressed to the hosts it serves (guarding against DNS rebinding),
takes style profiles by name only (no file paths from requests), and delivers exports as
downloads built in a temporary folder.
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from setsmith import __version__
from setsmith.analysis.liveset import LiveSet, compute_stats
from setsmith.graph.build import config_key, track_fingerprint
from setsmith.io.rekordbox_xml import ExportError, PlaylistSpec, export_playlists, load_collection
from setsmith.library import enrich, learned_config
from setsmith.model.collection import Collection, normalize_text
from setsmith.model.track import Track
from setsmith.scoring.suggest import suggest_next
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import (
    GeneratedSet,
    SetGenerationError,
    SetRequest,
    filter_pool,
    generate_set,
)
from setsmith.sets.report import render_markdown, set_timeline, set_to_dict, summary_line
from setsmith.store import Store, default_db_path
from setsmith.styles.profile import StyleError, StyleProfile, list_styles, load_style_file

STATIC_DIR = Path(__file__).parent / "static"
LOCAL_HOSTS = ["127.0.0.1", "localhost", "::1", "testserver"]
MAX_KEPT_SETS = 20
MAX_SEARCH_RESULTS = 20


class BuildBody(BaseModel):
    minutes: float | None = Field(default=60.0, gt=0, le=24 * 60)
    tracks: int | None = Field(default=None, ge=2, le=500)
    curve: str | None = None  # template name or custom points; default: style's, else journey
    style: str | None = None  # a key from /api/styles
    start_id: str | None = None
    bpm_min: float | None = None
    bpm_max: float | None = None
    genres: list[str] = []
    tags: list[str] = []
    exclude_ids: list[str] = []
    learned: bool = False
    name: str | None = Field(default=None, max_length=120)


class _State:
    def __init__(self, collection_path: Path, db: Path | None) -> None:
        self.collection_path = collection_path
        self.db = db
        self.collection: Collection = load_collection(collection_path)
        self.enrichment = enrich(self.collection, db)
        self.sets: OrderedDict[str, tuple[str, GeneratedSet]] = OrderedDict()
        self.lock = threading.Lock()

    def keep(self, name: str, gen: GeneratedSet) -> str:
        set_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.sets[set_id] = (name, gen)
            while len(self.sets) > MAX_KEPT_SETS:
                self.sets.popitem(last=False)
        return set_id

    def get(self, set_id: str) -> tuple[str, GeneratedSet]:
        with self.lock:
            found = self.sets.get(set_id)
        if found is None:
            raise HTTPException(404, "unknown set; build it again")
        return found


def _style(name: str | None) -> StyleProfile | None:
    """Styles by name only: a request must not make the server read arbitrary files."""
    if not name:
        return None
    entries = {e.key: e for e in list_styles()}
    entry = entries.get(name)
    if entry is None:
        raise HTTPException(400, f"unknown style {name!r}")
    try:
        return load_style_file(entry.path)
    except StyleError as exc:
        raise HTTPException(400, str(exc)) from exc


def _cfg(state: _State, learned: bool) -> ScoringConfig:
    if not learned:
        return DEFAULT_CONFIG
    cfg = learned_config(state.db)
    if cfg is None:
        raise HTTPException(400, "nothing learned yet; run 'setsmith learn' first")
    return cfg


def create_app(
    collection_path: Path, db: Path | None = None, extra_hosts: list[str] | None = None
) -> FastAPI:
    state = _State(collection_path, db)
    app = FastAPI(title="Setsmith", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOCAL_HOSTS, *(extra_hosts or [])])

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        col, e = state.collection, state.enrichment
        styles = []
        for entry in list_styles():
            try:
                p = load_style_file(entry.path)
            except StyleError:
                continue
            curve = p.energy_curve if isinstance(p.energy_curve, str) else "custom"
            styles.append(
                {"key": entry.key, "name": p.name, "bpm_band": p.bpm_band, "curve": curve}
            )
        return {
            "version": __version__,
            "collection": state.collection_path.name,
            "product": col.product,
            "tracks": len(col),
            "analysis": e.analysis.to_dict() if e.analysis else None,
            "my_tags": e.my_tags.to_dict() if e.my_tags else None,
            "curves": list(DEFAULT_CONFIG.sets.curves),
            "styles": styles,
            "learned_available": learned_config(state.db) is not None,
        }

    @app.get("/api/tracks")
    def tracks(
        q: str = Query("", max_length=200),
        limit: int = Query(10, ge=1, le=MAX_SEARCH_RESULTS),
    ) -> list[dict[str, Any]]:
        """Type-ahead search: every word must appear in artist or title (prefix matches
        first); falls back to fuzzy "Artist - Title" matching."""
        query = normalize_text(q)
        if not query:
            return []
        col = state.collection
        if q in col.tracks:
            return [{"match": 1.0, **col.tracks[q].summary()}]
        words = query.split()
        hits: list[tuple[tuple[int, int, str], Track]] = []
        for track in col.tracks.values():
            artist, title = col.normalized_names(track)
            text = f"{artist} {title}"
            if all(w in text for w in words):
                starts = 0 if text.startswith(query) or title.startswith(query) else 1
                hits.append(((starts, len(text), track.display), track))
        if hits:
            hits.sort(key=lambda h: h[0])
            return [{"match": 1.0, **t.summary()} for _, t in hits[:limit]]
        found = col.find(q)
        return [{"match": round(sc, 3), **t.summary()} for sc, t in found.candidates[:limit]]

    @app.get("/api/suggest")
    def suggest(
        track_id: str,
        top: int = Query(10, ge=1, le=50),
        energy_delta: float = Query(0.0, ge=-9, le=9),
        style: str | None = None,
        learned: bool = False,
    ) -> dict[str, Any]:
        seed = state.collection.tracks.get(track_id)
        if seed is None:
            raise HTTPException(404, f"no track with TrackID {track_id!r}")
        profile = _style(style)
        results = suggest_next(
            state.collection, seed, top=top, energy_target=energy_delta, style=profile,
            cfg=_cfg(state, learned),
        )  # fmt: skip
        return {
            "seed": seed.summary(),
            "style": profile.name if profile else None,
            "suggestions": [
                {
                    "rank": i,
                    "track": s.track.summary(),
                    "score": s.score.to_dict(),
                    "explain": s.score.explain(),
                    "style_fit": {"score": round(s.style_fit.score, 3), **s.style_fit.parts}
                    if s.style_fit
                    else None,
                }
                for i, s in enumerate(results, 1)
            ],
        }

    @app.post("/api/build")
    def build(body: BuildBody) -> dict[str, Any]:
        profile = _style(body.style)
        cfg = _cfg(state, body.learned)
        try:
            if body.curve:
                curve = parse_curve(body.curve, cfg)
            elif profile is not None:
                curve = profile.curve(cfg)
            else:
                curve = parse_curve("journey", cfg)
        except ValueError as exc:
            raise HTTPException(400, f"curve: {exc}") from exc
        genres = frozenset(g for item in body.genres for g in cfg.genre.normalize(item))
        request = SetRequest(
            curve=curve,
            minutes=body.minutes if body.tracks is None else None,
            track_count=body.tracks,
            start_id=body.start_id or None,
            bpm_min=body.bpm_min,
            bpm_max=body.bpm_max,
            genres=genres,
            exclude_ids=frozenset(body.exclude_ids),
            style=profile,
            tags=frozenset(t.casefold() for t in body.tags),
        )
        db_path = state.db or default_db_path()
        key = config_key(cfg, profile)
        try:
            with Store(db_path) as store:
                pool, _ = filter_pool(state.collection, request, cfg)
                cache = store.load_pair_scores(key, {track_fingerprint(t) for t in pool})
                gen, graph = generate_set(state.collection, request, cfg=cfg, cache=cache)
                store.save_pair_scores(key, graph.new_scores)
        except SetGenerationError as exc:
            raise HTTPException(422, str(exc)) from exc
        label = f"{profile.name} {curve.name}" if profile else curve.name
        name = body.name or f"{label} {gen.stats.track_count} tracks"
        set_id = state.keep(name, gen)
        data = set_to_dict(gen, name)
        for pos, entry in zip(data["positions"], gen.positions, strict=True):
            pos["explain"] = entry.transition.explain() if entry.transition else []
        return {
            "id": set_id,
            "set": data,
            "summary": summary_line(gen),
            "timeline": set_timeline(gen, cfg),
            "curve_targets": [round(p.target_energy, 2) for p in gen.positions],
        }

    @app.get("/api/sets/{set_id}/export")
    def export(set_id: str, alternates: bool = True) -> FileResponse:
        name, gen = state.get(set_id)
        playlists = [PlaylistSpec(name, tuple(gen.track_ids))]
        if alternates and gen.alternate_ids:
            playlists.append(PlaylistSpec(f"{name} (alternates)", tuple(gen.alternate_ids)))
        tmp = Path(tempfile.mkdtemp(prefix="setsmith-export-"))
        out = tmp / "setsmith.xml"
        try:
            export_playlists(state.collection_path, out, playlists)
        except ExportError as exc:
            raise HTTPException(500, str(exc)) from exc
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip() or "set"
        return FileResponse(
            out,
            media_type="application/xml",
            filename=f"{safe}.xml",
            background=BackgroundTask(_cleanup, tmp),
        )

    @app.get("/api/sets/{set_id}/report")
    def report(set_id: str) -> PlainTextResponse:
        name, gen = state.get(set_id)
        return PlainTextResponse(render_markdown(gen, name), media_type="text/markdown")

    @app.get("/api/livesets")
    def livesets() -> list[dict[str, Any]]:
        db_path = state.db or default_db_path()
        if not db_path.exists():
            return []
        with Store(db_path) as store:
            return [{"id": i, "name": n, "created_at": c} for i, n, c in store.list_livesets()]

    @app.get("/api/livesets/{liveset_id}")
    def liveset(liveset_id: int) -> dict[str, Any]:
        db_path = state.db or default_db_path()
        data = None
        if db_path.exists():
            with Store(db_path) as store:
                data = store.get_liveset(liveset_id)
        if data is None:
            raise HTTPException(404, f"no live set #{liveset_id}")
        ls = LiveSet.from_dict(data)
        return {"liveset": ls.to_dict(), "stats": compute_stats(ls).to_dict()}

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _cleanup(folder: Path) -> None:
    for child in folder.iterdir():
        child.unlink(missing_ok=True)
    folder.rmdir()

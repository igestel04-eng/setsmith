"""Setsmith command line interface."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from setsmith import __version__
from setsmith.analysis.apply import AnalysisCoverage, load_and_apply
from setsmith.analysis.audio import (
    AnalysisError,
    AnalysisJob,
    AnalysisOptions,
    TrackAnalysis,
    analyze_many,
    default_workers,
    has_module,
    location_to_path,
)
from setsmith.analysis.learn import (
    history_pairs,
    learn,
    move_table,
    tempo_table,
)
from setsmith.analysis.liveset import LiveSet, analyze_liveset, compute_stats, draft_profile
from setsmith.analysis.tracklist import load_tracklist, match_tracklist
from setsmith.discovery.discover import ATTRIBUTIONS as DISCOVERY_ATTRIBUTIONS
from setsmith.discovery.discover import DiscoveryUnavailable, make_sources
from setsmith.discovery.discover import discover as run_discovery
from setsmith.discovery.http import ServiceError
from setsmith.discovery.keys import SERVICES, key_source, remove_key, set_key
from setsmith.graph.build import config_key, track_fingerprint
from setsmith.io.rekordbox_db import (
    KEY_ENV,
    RekordboxDbError,
    copy_master_db,
    default_master_db,
    read_master_db,
    rekordbox_running,
)
from setsmith.io.rekordbox_db import normalize_path as normalize_file_path
from setsmith.io.rekordbox_xml import (
    ExportError,
    PlaylistSpec,
    RekordboxXMLError,
    export_playlists,
    load_collection,
)
from setsmith.keys.camelot import classify_move, parse_key
from setsmith.library import Enrichment, enrich, learned_config
from setsmith.model.collection import Collection, NodeType
from setsmith.model.track import Track
from setsmith.scoring.suggest import Suggestion, suggest_next
from setsmith.scoring.transition import TransitionType, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG, ScoringConfig
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import GeneratedSet, SetGenerationError, SetRequest, filter_pool
from setsmith.sets.generate import generate_set as run_generation
from setsmith.sets.report import (
    render_markdown,
    set_comments,
    set_to_dict,
    style_line,
    summary_line,
)
from setsmith.store import Store, default_db_path
from setsmith.styles.profile import (
    StyleError,
    StyleProfile,
    list_styles,
    load_style,
    load_style_file,
    user_styles_dir,
)

app = typer.Typer(
    help="Transition-aware DJ set builder for Rekordbox XML exports. Reads only; never "
    "modifies your Rekordbox library or audio files.",
    no_args_is_help=True,
    add_completion=False,
)
out = Console()
err = Console(stderr=True)

EXIT_LOAD_ERROR = 1
EXIT_TRACK_NOT_FOUND = 2
EXIT_BUILD_ERROR = 3

CollectionArg = Annotated[
    Path,
    typer.Argument(
        help="Rekordbox XML export (File > Export Collection in xml format).",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
NoAnalysisOpt = Annotated[
    bool,
    typer.Option("--no-analysis", help="Ignore stored audio analysis; use Rekordbox tags only."),
]
LearnedOpt = Annotated[
    bool,
    typer.Option(
        "--learned",
        help="Use key-move and tempo weights learned from your own sets ('setsmith learn').",
    ),
]
StyleOpt = Annotated[
    str | None,
    typer.Option(
        "--style",
        "-s",
        help="Style profile: a name from 'setsmith styles list' or a path to a .json file.",
    ),
]
DbOpt = Annotated[
    Path | None,
    typer.Option(
        "--db",
        help=f"SQLite store for the score cache and feedback. Default: {default_db_path()}",
        dir_okay=False,
    ),
]


def _version(value: bool) -> None:
    if value:
        out.print(f"setsmith {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Setsmith: score transitions, suggest next tracks, build sets."""


def _load(path: Path) -> Collection:
    try:
        return load_collection(path)
    except (RekordboxXMLError, OSError) as exc:
        err.print(f"[red]Could not read {path}:[/red] {exc}")
        raise typer.Exit(EXIT_LOAD_ERROR) from exc


def _load_analyzed(
    path: Path, db: Path | None, use_analysis: bool
) -> tuple[Collection, Enrichment]:
    """Load a collection and apply stored audio analysis and imported My Tags, if any."""
    col = _load(path)
    return col, enrich(col, db, use_analysis)


def _scoring_config(use_learned: bool, db: Path | None) -> ScoringConfig:
    """Default weights, or weights blended with learned preferences when asked."""
    if not use_learned:
        return DEFAULT_CONFIG
    cfg = learned_config(db)
    if cfg is None:
        raise typer.BadParameter(
            "nothing learned yet; run 'setsmith learn <xml>' first", param_hint="--learned"
        )
    return cfg


def _load_style_opt(name: str | None) -> StyleProfile | None:
    if name is None:
        return None
    try:
        return load_style(name)
    except StyleError as exc:
        raise typer.BadParameter(str(exc), param_hint="--style") from exc


def _emit_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _fail_track(
    message: str, candidates: list[tuple[float, Track]], as_json: bool, id_flag: str = "--id"
) -> NoReturn:
    if as_json:
        _emit_json(
            {
                "error": message,
                "candidates": [
                    {"match": round(score, 3), **t.summary()} for score, t in candidates
                ],
            }
        )
    else:
        err.print(f"[red]{message}[/red]")
        if candidates:
            table = Table(title="Did you mean", title_justify="left")
            table.add_column("TrackID", style="cyan")
            table.add_column("Track")
            table.add_column("BPM", justify="right")
            table.add_column("Key")
            table.add_column("Match", justify="right")
            for score, t in candidates:
                table.add_row(t.id, t.display, _bpm(t), _key(t), f"{score:.2f}")
            err.print(table)
            err.print(f"Pick one with [bold]{id_flag} <TrackID>[/bold].")
    raise typer.Exit(EXIT_TRACK_NOT_FOUND)


def _resolve_track(
    collection: Collection,
    query: str | None,
    track_id: str | None,
    as_json: bool,
    *,
    flags: tuple[str, str] = ("--track", "--id"),
    required: bool = True,
) -> Track | None:
    """Find a track by TrackID or "Artist - Title"; exits with candidates when ambiguous."""
    query_flag, id_flag = flags
    if track_id is not None:
        found = collection.tracks.get(track_id)
        if found is None:
            _fail_track(f"No track with TrackID {track_id!r}", [], as_json, id_flag)
        return found
    if not query:
        if not required:
            return None
        raise typer.BadParameter(
            f"give a track with {query_flag} or {id_flag}", param_hint=query_flag
        )
    match = collection.find(query)
    if match.track is not None:
        return match.track
    if match.ambiguous:
        _fail_track(f"{query!r} matches several tracks", match.candidates, as_json, id_flag)
    _fail_track(f"No track matches {query!r}", [], as_json, id_flag)


def _resolve_seed(
    collection: Collection, track: str | None, track_id: str | None, as_json: bool
) -> Track:
    found = _resolve_track(collection, track, track_id, as_json)
    assert found is not None  # required=True exits otherwise
    return found


def _bpm(t: Track) -> str:
    return f"{t.bpm:g}" if t.bpm else "-"


def _key(t: Track) -> str:
    return t.camelot or ("?" if t.key_raw else "-")


def _energy(t: Track) -> str:
    return f"{t.energy:g}" if t.energy is not None else "-"


def _score_text(total: float) -> Text:
    d = DEFAULT_CONFIG.display
    style = "green" if total >= d.good_score else "yellow" if total >= d.ok_score else "red"
    return Text(f"{total:.0f}", style=f"bold {style}")


def _pct(score: float) -> str:
    return f"{100 * score:.0f}"


def _seed_line(t: Track) -> Text:
    key = parse_key(t.camelot)
    parts = [
        f"{_bpm(t)} BPM",
        f"{t.camelot} ({key.name})" if key else f"key {t.key_raw or 'unknown'}",
        f"E{_energy(t)}",
        t.genre or "no genre",
    ]
    line = Text("Seed  ", style="bold")
    line.append(t.display, style="bold cyan")
    line.append("  " + " · ".join(parts), style="dim")
    return line


def _suggest_table(suggestions: list[Suggestion], wide: bool) -> Table:
    """Full table with component columns when wide, else a compact one with flags under titles."""
    table = Table(pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Track", overflow="fold", ratio=1)
    table.add_column("BPM", justify="right")
    table.add_column("Key")
    table.add_column("E", justify="right")
    table.add_column("Score", justify="right")
    if wide:
        for name in ("Harm", "Tempo", "Enrg", "Genre", "Extra"):
            table.add_column(name, justify="right", style="dim")
    styled = any(s.style_fit is not None for s in suggestions)
    if styled:
        table.add_column("Style", justify="right")
    table.add_column("Transition", no_wrap=True)
    if wide:
        table.add_column("Flags", style="yellow", overflow="fold", ratio=1)

    for rank, s in enumerate(suggestions, 1):
        flags = ", ".join(s.score.flags)
        track_cell = Text(s.track.display)
        if flags and not wide:
            track_cell.append(f"\n{flags}", style="yellow")
        row: list[str | Text] = [
            str(rank),
            track_cell,
            _bpm(s.track),
            _key(s.track),
            _energy(s.track),
            _score_text(s.score.total),
        ]
        if wide:
            row += [_pct(c.score) for c in s.score.components.values()]
        if styled:
            row.append(_pct(s.style_fit.score) if s.style_fit else "-")
        row.append(f"{s.score.suggested_type.value} {s.score.suggested_length_bars}b")
        if wide:
            row.append(flags)
        table.add_row(*row)
    return table


@app.command()
def suggest(
    collection: CollectionArg,
    track: Annotated[
        str | None,
        typer.Option("--track", "-t", help='Seed track: "Artist - Title", or the title alone.'),
    ] = None,
    track_id: Annotated[
        str | None, typer.Option("--id", help="Seed track by Rekordbox TrackID.")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", min=1, help="Number of suggestions.")] = 10,
    energy_delta: Annotated[
        float,
        typer.Option(
            "--energy-delta",
            help="Desired energy change on the 1-10 scale (e.g. 2 to build, -1 to ease off).",
        ),
    ] = DEFAULT_CONFIG.energy.default_target_delta,
    explain: Annotated[
        bool, typer.Option("--explain", "-x", help="Show the reason behind every component score.")
    ] = False,
    style: StyleOpt = None,
    learned: LearnedOpt = False,
    db: DbOpt = None,
    no_analysis: NoAnalysisOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Suggest the best tracks to play after a seed track, with a score breakdown.

    With --style, key moves and genres are weighted by the profile and each suggestion
    shows its style fit (0-100); the fit is shown, not added to the score.
    """
    profile = _load_style_opt(style)
    col, _ = _load_analyzed(collection, db, not no_analysis)
    seed = _resolve_seed(col, track, track_id, as_json)
    cfg = _scoring_config(learned, db)
    results = suggest_next(col, seed, top=top, energy_target=energy_delta, style=profile, cfg=cfg)

    if as_json:
        _emit_json(
            {
                "seed": seed.summary(),
                "energy_target": energy_delta,
                "style": profile.name if profile else None,
                "suggestions": [
                    {
                        "rank": i,
                        "track": s.track.summary(),
                        "score": s.score.to_dict(),
                        "style_fit": {"score": round(s.style_fit.score, 3), **s.style_fit.parts}
                        if s.style_fit
                        else None,
                    }
                    for i, s in enumerate(results, 1)
                ],
            }
        )
        return

    out.print(_seed_line(seed))
    if profile is not None:
        out.print(f"[dim]Style: {profile.name} (a profile, not endorsed by the artists)[/dim]")
    if not results:
        out.print("No other tracks in the collection.")
        return
    out.print(
        _suggest_table(results, wide=out.width >= DEFAULT_CONFIG.display.wide_table_min_width)
    )
    if explain:
        for rank, s in enumerate(results, 1):
            out.print(f"\n[bold]{rank}. {s.track.display}[/bold]  [dim]{s.score.total:.1f}[/dim]")
            for line in s.score.explain():
                out.print(f"   {line}", highlight=False)
            if s.style_fit is not None:
                fit = s.style_fit
                out.print(
                    f"   {'style fit':<12} {fit.score:4.2f}  {fit.describe()}", highlight=False
                )
    else:
        out.print("[dim]Scores are 0-100. Add --explain for the reasoning behind each score.[/dim]")


@app.command()
def info(
    collection: CollectionArg,
    db: DbOpt = None,
    no_analysis: NoAnalysisOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Summarize a collection: metadata coverage, keys, genres, analysis and warnings."""
    col, enrichment = _load_analyzed(collection, db, not no_analysis)
    analysis = enrichment.analysis
    tracks = list(col.tracks.values())
    n = len(tracks)
    playlists = (
        sum(1 for _, node in col.playlists.walk() if node.type == NodeType.PLAYLIST)
        if col.playlists
        else 0
    )
    coverage = {
        "bpm": sum(1 for t in tracks if t.bpm),
        "key": sum(1 for t in tracks if t.camelot),
        "energy": sum(1 for t in tracks if t.energy is not None),
        "genre": sum(1 for t in tracks if t.genre),
        "rating": sum(1 for t in tracks if t.rating),
        "cues": sum(1 for t in tracks if t.cues),
        "variable_tempo": sum(1 for t in tracks if t.variable_tempo),
        "intro_outro": sum(1 for t in tracks if t.intro_bars is not None),
    }
    keys = Counter(t.camelot for t in tracks if t.camelot)
    genres = Counter(g for t in tracks for g in DEFAULT_CONFIG.genre.normalize(t.genre))

    if as_json:
        _emit_json(
            {
                "path": str(collection),
                "product": col.product,
                "tracks": n,
                "playlists": playlists,
                "coverage": coverage,
                "keys": dict(keys.most_common()),
                "genres": dict(genres.most_common()),
                "analysis": analysis.to_dict() if analysis else None,
                "my_tags": enrichment.my_tags.to_dict() if enrichment.my_tags else None,
                "rekordbox_import": enrichment.rekordbox,
                "warnings": col.warnings,
            }
        )
        return

    out.print(f"[bold]{collection.name}[/bold]  [dim]{col.product}[/dim]")
    out.print(f"{n} tracks, {playlists} playlists")
    table = Table("Field", "Tracks", "Share", title="Metadata coverage", title_justify="left")
    for name, count in coverage.items():
        table.add_row(name, str(count), f"{100 * count / n:.0f}%" if n else "-")
    out.print(table)
    if analysis is not None:
        _print_analysis_coverage(analysis)
    if enrichment.rekordbox and enrichment.my_tags is not None:
        rb = enrichment.rekordbox
        out.print(
            f"[bold]Rekordbox My Tags[/bold]  {enrichment.my_tags.tagged} tracks tagged, "
            f"{rb['sessions']} history sessions (imported {rb['imported_at'][:10]})"
        )
    if genres:
        top_genres = ", ".join(f"{g} ({c})" for g, c in genres.most_common(8))
        out.print(f"[bold]Top genres[/bold]  {top_genres}")
    if keys:
        out.print(
            f"[bold]Top keys[/bold]  {', '.join(f'{k} ({c})' for k, c in keys.most_common(8))}"
        )
    shown = DEFAULT_CONFIG.display.max_warnings_shown
    for warning in col.warnings[:shown]:
        out.print(f"[yellow]warning:[/yellow] {warning}", highlight=False)
    if len(col.warnings) > shown:
        out.print(
            f"[yellow]... and {len(col.warnings) - shown} more warnings (see --json)[/yellow]"
        )


REKORDBOX_IMPORT_STEPS = (
    "To load it in Rekordbox: Preferences > View > enable 'rekordbox xml'; "
    "Preferences > Advanced > Database > rekordbox xml > choose this file; then in the "
    "sidebar open rekordbox xml > Setsmith, right-click the playlist, Import Playlist."
)


def _set_table(gen: GeneratedSet, wide: bool) -> Table:
    table = Table(pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Track", overflow="fold", ratio=1)
    table.add_column("BPM", justify="right")
    table.add_column("Key")
    table.add_column("E (tgt)", justify="right")
    table.add_column("Next", justify="right")
    table.add_column("Transition", no_wrap=True)
    if wide:
        table.add_column("Flags", style="yellow", overflow="fold", ratio=1)
    for p in gen.positions:
        tr = p.transition
        flags = ", ".join(tr.flags) if tr else ""
        track_cell = Text(p.track.display)
        if flags and not wide:
            track_cell.append(f"\n{flags}", style="yellow")
        row: list[str | Text] = [
            str(p.index + 1),
            track_cell,
            _bpm(p.track),
            _key(p.track),
            f"{_energy(p.track)} ({p.target_energy:.1f})",
            _score_text(tr.total) if tr else Text(""),
            f"{tr.suggested_type.value} {tr.suggested_length_bars}b" if tr else "end",
        ]
        if wide:
            row.append(flags)
        table.add_row(*row)
    return table


def _print_alternates(gen: GeneratedSet) -> None:
    rows = [p for p in gen.positions if p.alternates]
    if not rows:
        return
    out.print("[bold]Alternates[/bold] [dim](fit both neighbors; swap in to read the crowd)[/dim]")
    for p in rows:
        alts = "  ·  ".join(
            f"{a.track.display} [dim]{_bpm(a.track)} {_key(a.track)} E{_energy(a.track)}[/dim]"
            for a in p.alternates
        )
        out.print(f"  [dim]{p.index + 1:>2}[/dim]  {alts}", highlight=False)


def _resolve_excludes(collection: Collection, items: list[str]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        if item in collection.tracks:
            ids.add(item)
            continue
        match = collection.find(item)
        if match.track is None:
            reason = "is ambiguous" if match.ambiguous else "matches nothing"
            raise typer.BadParameter(f"{item!r} {reason}; use a TrackID", param_hint="--exclude")
        ids.add(match.track.id)
    return ids


@app.command()
def build(
    collection: CollectionArg,
    minutes: Annotated[
        float | None, typer.Option("--minutes", "-m", min=1, help="Set length in minutes.")
    ] = None,
    tracks: Annotated[
        int | None, typer.Option("--tracks", "-n", min=2, help="Set length in tracks.")
    ] = None,
    curve: Annotated[
        str | None,
        typer.Option(
            "--curve",
            "-c",
            help="Energy curve: warm_up, peak_time, closing, journey, or custom points "
            "such as '3,5,8,6' or '0:3,0.6:9,1:5' (default: the style's curve, else journey).",
        ),
    ] = None,
    style: StyleOpt = None,
    learned: LearnedOpt = False,
    start: Annotated[
        str | None, typer.Option("--start", help='Opening track: "Artist - Title".')
    ] = None,
    start_id: Annotated[
        str | None, typer.Option("--start-id", help="Opening track by TrackID.")
    ] = None,
    bpm_min: Annotated[float | None, typer.Option(help="Lowest BPM allowed.")] = None,
    bpm_max: Annotated[float | None, typer.Option(help="Highest BPM allowed.")] = None,
    genre: Annotated[
        list[str] | None,
        typer.Option("--genre", "-g", help="Only these genres (repeat for several)."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-x", help='Leave out a TrackID or "Artist - Title" (repeat).'),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help="Only tracks with this Rekordbox My Tag (repeat for any of several; "
            "needs 'setsmith rekordbox import').",
        ),
    ] = None,
    artist_gap: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Tracks before an artist may repeat "
            f"(default {DEFAULT_CONFIG.sets.artist_gap_tracks}; 0 disables).",
        ),
    ] = None,
    max_drift: Annotated[
        float | None,
        typer.Option(
            min=0,
            help="Tempo range (BPM) allowed over the set before a penalty "
            f"(default {DEFAULT_CONFIG.sets.max_tempo_drift_bpm:g}).",
        ),
    ] = None,
    beam: Annotated[
        int | None,
        typer.Option(min=1, help=f"Beam width (default {DEFAULT_CONFIG.sets.beam_width})."),
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="Playlist name (default: curve, length and time).")
    ] = None,
    out_path: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            dir_okay=False,
            help="Write the set as a playlist into this new Rekordbox XML file.",
        ),
    ] = None,
    alternates_playlist: Annotated[
        bool,
        typer.Option(help="Also export the alternates as a second playlist."),
    ] = True,
    report: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Write the full breakdown (.md, or .json) here."),
    ] = None,
    write_comments: Annotated[
        bool,
        typer.Option(
            help="Put position and score into track Comments in the exported XML. Rekordbox "
            "may copy these into your library's Comments when importing.",
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Replace --out even if it was not written by Setsmith.")
    ] = False,
    db: DbOpt = None,
    no_analysis: NoAnalysisOpt = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Don't read or write the pair-score cache.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Build a set that follows an energy curve, and optionally export it to Rekordbox XML."""
    col, _ = _load_analyzed(collection, db, not no_analysis)
    start_track = _resolve_track(
        col, start, start_id, as_json, flags=("--start", "--start-id"), required=False
    )
    profile = _load_style_opt(style)
    try:
        if curve is not None:
            energy_curve = parse_curve(curve)
        elif profile is not None:
            energy_curve = profile.curve()
        else:
            energy_curve = parse_curve("journey")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--curve") from exc
    genres = frozenset(g for item in genre or [] for g in DEFAULT_CONFIG.genre.normalize(item))
    if minutes is None and tracks is None:
        minutes = 60.0

    request = SetRequest(
        curve=energy_curve,
        minutes=minutes if tracks is None else None,
        track_count=tracks,
        start_id=start_track.id if start_track else None,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        genres=genres,
        exclude_ids=frozenset(_resolve_excludes(col, exclude or [])),
        artist_gap=artist_gap,
        max_tempo_drift_bpm=max_drift,
        beam_width=beam,
        style=profile,
        tags=frozenset(t.casefold() for t in tag or []),
    )

    store = None if no_cache else Store(db)
    cfg = _scoring_config(learned, db)
    cfg_key = config_key(cfg, profile)
    try:
        cache = None
        if store is not None:
            pool, _ = filter_pool(col, request, cfg)
            cache = store.load_pair_scores(cfg_key, {track_fingerprint(t) for t in pool})
        generated, graph = run_generation(col, request, cfg=cfg, cache=cache)
        if store is not None:
            store.save_pair_scores(cfg_key, graph.new_scores)
    except SetGenerationError as exc:
        err.print(f"[red]Could not build a set:[/red] {exc}")
        raise typer.Exit(EXIT_BUILD_ERROR) from exc
    finally:
        if store is not None:
            store.close()

    length = f"{tracks} tracks" if tracks else f"{minutes:g}min"
    label = f"{profile.name} {energy_curve.name}" if profile else energy_curve.name
    set_name = name or f"{label} {length} {datetime.now():%Y-%m-%d %H.%M}"

    exported = None
    if out_path is not None:
        playlists = [PlaylistSpec(set_name, tuple(generated.track_ids))]
        if alternates_playlist and generated.alternate_ids:
            playlists.append(
                PlaylistSpec(f"{set_name} (alternates)", tuple(generated.alternate_ids))
            )
        comments = set_comments(generated, set_name) if write_comments else None
        try:
            exported = export_playlists(
                collection, out_path, playlists, comments=comments, overwrite=force
            )
        except ExportError as exc:
            err.print(f"[red]Export failed:[/red] {exc}")
            raise typer.Exit(EXIT_BUILD_ERROR) from exc

    data = set_to_dict(generated, set_name)
    if report is not None:
        text = (
            json.dumps(data, indent=2, ensure_ascii=False)
            if report.suffix.lower() == ".json"
            else render_markdown(generated, set_name)
        )
        report.write_text(text + "\n", encoding="utf-8")

    if as_json:
        data["export"] = (
            {
                "path": str(exported.path),
                "playlists": exported.written,
                "kept_playlists": exported.kept,
                "tracks": exported.track_count,
            }
            if exported
            else None
        )
        data["report"] = str(report) if report else None
        _emit_json(data)
        return

    out.print(Text(set_name, style="bold cyan"))
    out.print(f"[dim]{summary_line(generated)}[/dim]")
    if generated.stats.style is not None:
        out.print(f"[dim]{style_line(generated)}[/dim]", highlight=False)
    for warning in generated.warnings:
        out.print(f"[yellow]warning:[/yellow] {warning}", highlight=False)
    out.print(_set_table(generated, wide=out.width >= DEFAULT_CONFIG.display.wide_table_min_width))
    _print_alternates(generated)
    if report is not None:
        out.print(f"Report written to [bold]{report}[/bold]")
    if exported is not None:
        kept = f" (kept {len(exported.kept)} earlier Setsmith playlists)" if exported.kept else ""
        out.print(
            f"Wrote {', '.join(repr(n) for n in exported.written)} to "
            f"[bold]{exported.path}[/bold]{kept}."
        )
        out.print(f"[dim]{REKORDBOX_IMPORT_STEPS}[/dim]")


# ---------------------------------------------------------------- analyze


class StructureBackend(StrEnum):
    GRID = "grid"
    ALLIN1 = "allin1"


def _print_analysis_coverage(cov: AnalysisCoverage) -> None:
    parts = [f"{cov.analyzed} analyzed"]
    if cov.stale:
        parts.append(f"{cov.stale} changed since analysis")
    if cov.not_analyzed:
        parts.append(f"{cov.not_analyzed} not analyzed")
    if cov.missing_file:
        parts.append(f"{cov.missing_file} files not found")
    out.print(f"[bold]Audio analysis[/bold]  {', '.join(parts)}")
    checked = cov.key_agree + cov.key_related + cov.key_disagree
    if checked:
        out.print(
            f"  keys vs tags: {cov.key_agree} agree, {cov.key_related} adjacent/relative, "
            f"{cov.key_disagree} disagree"
            + (f"; {cov.key_from_audio} untagged filled from audio" if cov.key_from_audio else "")
        )


def _select_tracks(
    col: Collection, playlist: str | None, track: str | None, track_id: str | None, as_json: bool
) -> list[Track]:
    if track or track_id:
        found = _resolve_track(col, track, track_id, as_json)
        return [found] if found else []
    if playlist is None:
        return list(col.tracks.values())
    if col.playlists is None:
        raise typer.BadParameter("the collection has no playlists", param_hint="--playlist")
    matches = [
        node
        for path, node in col.playlists.walk()
        if node.type == NodeType.PLAYLIST and playlist in ("/".join(path[1:]), node.name)
    ]
    if len(matches) != 1:
        reason = "matches several playlists; use Folder/Name" if matches else "not found"
        raise typer.BadParameter(f"{playlist!r} {reason}", param_hint="--playlist")
    return col.playlist_tracks(matches[0])


@app.command()
def analyze(
    collection: CollectionArg,
    playlist: Annotated[
        str | None,
        typer.Option("--playlist", "-p", help='Only this playlist: "Name" or "Folder/Name".'),
    ] = None,
    track: Annotated[
        str | None, typer.Option("--track", "-t", help='Only this track: "Artist - Title".')
    ] = None,
    track_id: Annotated[str | None, typer.Option("--id", help="Only this TrackID.")] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Analyze at most this many new tracks.")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Parallel processes.")
    ] = default_workers(),
    force: Annotated[
        bool, typer.Option(help="Re-analyze files even if a current result is stored.")
    ] = False,
    key_profile: Annotated[
        str, typer.Option(help="Essentia key profile (edma, bgate, edmm, temperley, ...).")
    ] = DEFAULT_CONFIG.analysis.essentia_key_profile,
    structure: Annotated[
        StructureBackend,
        typer.Option(
            help="Intro/outro detection: 'grid' (beat grid + harmonic energy) or 'allin1' "
            "(needs allin1 installed separately)."
        ),
    ] = StructureBackend.GRID,
    lufs: Annotated[
        bool,
        typer.Option(help="Also measure EBU R128 loudness (Essentia; slower, informational)."),
    ] = False,
    no_essentia: Annotated[
        bool, typer.Option("--no-essentia", help="Use librosa only, even if Essentia is installed.")
    ] = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Analyze local audio files: key, loudness, energy and intro/outro length.

    Reads the files at each track's Location; never modifies them. Results are stored in
    the Setsmith database and used automatically by suggest, build and info.
    """
    if not has_module("librosa"):
        err.print(
            "[red]Audio analysis needs the audio extra.[/red] Install it with: "
            "uv sync --extra audio  (add --extra essentia for EDM key profiles)"
        )
        raise typer.Exit(EXIT_LOAD_ERROR)
    col = _load(collection)
    selected = _select_tracks(col, playlist, track, track_id, as_json)
    cfg = DEFAULT_CONFIG.analysis
    use_essentia = not no_essentia and has_module("essentia")
    options = AnalysisOptions(
        essentia=use_essentia, key_profile=key_profile, structure=structure.value, lufs=lufs
    )

    missing: list[Track] = []
    files: dict[str, tuple[str, float, int]] = {}
    for t in selected:
        path = location_to_path(t.location)
        try:
            st = path.stat() if path else None
        except OSError:
            st = None
        if path is None or st is None:
            missing.append(t)
        else:
            files[t.id] = (str(path), st.st_mtime, st.st_size)

    failures: list[tuple[Track, str]] = []
    done = 0
    with Store(db) as store:
        current = {} if force else store.load_analyses(files.values(), cfg.version)
        todo = [t for t in selected if t.id in files and files[t.id][0] not in current]
        if limit is not None:
            todo = todo[:limit]
        jobs = [AnalysisJob(t.id, files[t.id][0], tuple(t.beat_grid), t.bpm) for t in todo]
        by_id = {t.id: t for t in todo}

        backend = f"Essentia ({key_profile}) + librosa" if use_essentia else "librosa"
        if not as_json:
            out.print(
                f"{len(selected)} selected: {len(jobs)} to analyze, "
                f"{sum(1 for path, _, _ in files.values() if path in current)} already current, "
                f"{len(missing)} files not found. Backend: {backend}; structure: "
                f"{structure.value}; {min(workers, max(1, len(jobs)))} worker(s)."
            )
        results = analyze_many(jobs, options, workers=min(workers, max(1, len(jobs))))
        try:
            if as_json or not jobs:
                for job, result in results:
                    done += _record(store, by_id[job.track_id], result, failures)
            else:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    TextColumn,
                    TimeRemainingColumn,
                )

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    console=out,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Analyzing", total=len(jobs))
                    for job, result in results:
                        done += _record(store, by_id[job.track_id], result, failures)
                        progress.update(
                            task, advance=1, description=f"Analyzing {by_id[job.track_id].display}"
                        )
        except KeyboardInterrupt:
            err.print(
                f"[yellow]Stopped.[/yellow] {done} results were saved; run again to continue."
            )
            raise typer.Exit(130) from None

        coverage = load_and_apply(col, store)

    disagreements = [
        t
        for t in selected
        if t.detected_camelot and t.key_raw and t.key_confidence < cfg.key_confidence_related
    ]
    if as_json:
        _emit_json(
            {
                "selected": len(selected),
                "analyzed": done,
                "failed": [{"id": t.id, "track": t.display, "error": e} for t, e in failures],
                "missing_files": [{"id": t.id, "location": t.location} for t in missing],
                "backend": {
                    "key": "essentia" if use_essentia else "librosa",
                    "key_profile": key_profile if use_essentia else "krumhansl-kessler",
                    "structure": structure.value,
                },
                "coverage": coverage.to_dict(),
                "key_disagreements": [
                    {
                        "id": t.id,
                        "track": t.display,
                        "tag": t.camelot,
                        "detected": t.detected_camelot,
                    }
                    for t in disagreements
                ],
            }
        )
        return

    out.print(f"Analyzed {done} file(s).")
    shown = DEFAULT_CONFIG.display.max_warnings_shown
    for t, error in failures[:shown]:
        out.print(f"[yellow]failed:[/yellow] {t.display}: {error}", highlight=False)
    if len(failures) > shown:
        out.print(f"[yellow]... and {len(failures) - shown} more (see --json)[/yellow]")
    for t in missing[:shown]:
        out.print(f"[yellow]not found:[/yellow] {t.display} ({t.location})", highlight=False)
    _print_analysis_coverage(coverage)
    if disagreements:
        table = Table(
            "Track",
            "Tag",
            "Audio",
            "Move",
            title="Key disagreements (scored at lower confidence)",
            title_justify="left",
            pad_edge=False,
        )
        for t in disagreements[:shown]:
            tag, det = parse_key(t.camelot), parse_key(t.detected_camelot)
            move = classify_move(tag, det).value if tag and det else "-"
            table.add_row(t.display, t.camelot or "-", t.detected_camelot or "-", move)
        out.print(table)


def _record(
    store: Store,
    track: Track,
    result: TrackAnalysis | AnalysisError,
    failures: list[tuple[Track, str]],
) -> int:
    if isinstance(result, AnalysisError):
        failures.append((track, str(result)))
        return 0
    store.save_analysis(result)
    return 1


# ---------------------------------------------------------------- styles

styles_app = typer.Typer(
    help="DJ style profiles: musical parameters inspired by a DJ's public output. "
    "Profiles imply no endorsement by the artists; edit them freely.",
    no_args_is_help=True,
)
app.add_typer(styles_app, name="styles")


@styles_app.command("list")
def styles_list(as_json: JsonOpt = False) -> None:
    """List available style profiles (built-in and your own)."""
    entries = []
    for entry in list_styles():
        try:
            profile: StyleProfile | None = load_style_file(entry.path)
            problem = ""
        except StyleError as exc:
            profile, problem = None, str(exc)
        entries.append((entry, profile, problem))
    if as_json:
        _emit_json(
            [
                {
                    "key": e.key,
                    "name": p.name if p else None,
                    "path": str(e.path),
                    "builtin": e.builtin,
                    "error": problem or None,
                }
                for e, p, problem in entries
            ]
        )
        return
    table = Table("Key", "Name", "BPM", "Curve", "Source", pad_edge=False)
    for entry, profile, problem in entries:
        source = "built-in" if entry.builtin else str(entry.path)
        if profile is None:
            table.add_row(entry.key, Text(problem, style="red"), "", "", source)
            continue
        lo, hi = profile.bpm_band
        curve = profile.energy_curve if isinstance(profile.energy_curve, str) else "custom"
        table.add_row(entry.key, profile.name, f"{lo:g}-{hi:g}", curve, source)
    out.print(table)
    out.print(
        f"[dim]Your own profiles go in {user_styles_dir()} (see 'setsmith styles copy').[/dim]"
    )


@styles_app.command("show")
def styles_show(
    name: Annotated[str, typer.Argument(help="Profile name or path to a .json file.")],
    as_json: JsonOpt = False,
) -> None:
    """Show a profile's parameters."""
    profile = _load_style_opt(name)
    assert profile is not None
    if as_json:
        _emit_json(profile.model_dump(mode="json"))
        return
    out.print(f"[bold]{profile.name}[/bold]  [dim](a profile, not endorsed by the artists)[/dim]")
    out.print(profile.description, highlight=False)
    lo, hi = profile.bpm_band
    rows = [
        ("BPM", f"{lo:g}-{hi:g} (preferred {profile.bpm_preferred:g}, drift up to "
                f"{profile.max_tempo_drift_bpm:g})"),
        ("Keys", f"minor {profile.key_mode_preference.minor:g} / major "
                 f"{profile.key_mode_preference.major:g}"),
        ("Key moves", ", ".join(f"{k.value} {v:g}" for k, v in profile.allowed_key_moves.items())),
        ("Energy curve", str(profile.energy_curve)),
        ("Genres", ", ".join(f"{g} {w:g}" for g, w in profile.genre_weights.items())),
        ("Vocal share", f"{profile.vocal_density:g}"),
        ("Blend lengths", ", ".join(
            f"{b} bars {p:g}" for b, p in profile.transition_length_bars.items())),
        ("Transitions", ", ".join(f"{t} {p:g}" for t, p in profile.transition_type_mix.items())),
        ("Artists", ", ".join(profile.reference_artists)),
        ("Labels", ", ".join(profile.reference_labels)),
        ("Notes", profile.notes),
        ("Sources", "; ".join(profile.sources)),
    ]  # fmt: skip
    table = Table(show_header=False, pad_edge=False, box=None)
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    for label, value in rows:
        if value:
            table.add_row(label, value)
    out.print(table)


@styles_app.command("copy")
def styles_copy(
    name: Annotated[str, typer.Argument(help="Profile to copy.")],
    new_name: Annotated[
        str | None, typer.Argument(help="Key for the copy (default: same as the original).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing profile file.")] = False,
) -> None:
    """Copy a profile into your styles folder to edit it. Same-named copies override built-ins."""
    entries = {e.key: e for e in list_styles()}
    source = entries.get(name)
    if source is None:
        raise typer.BadParameter(
            f"no style profile named {name!r} (available: {', '.join(entries) or 'none'})",
            param_hint="NAME",
        )
    target = user_styles_dir() / f"{new_name or name}.json"
    if target.exists() and not force:
        err.print(f"[red]{target} already exists.[/red] Edit it, or pass --force to replace it.")
        raise typer.Exit(EXIT_LOAD_ERROR)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.path.read_text(encoding="utf-8"), encoding="utf-8")
    out.print(f"Copied to [bold]{target}[/bold]. Edit it, then use --style {target.stem}.")


# ---------------------------------------------------------------- rekordbox master.db

rekordbox_app = typer.Typer(
    help="Optional, read-only: import My Tags and play history from a copy of Rekordbox's "
    "master.db. Back up your library first (Rekordbox: File > Library > Backup Library).",
    no_args_is_help=True,
)
app.add_typer(rekordbox_app, name="rekordbox")

BACKUP_WARNING = (
    "Setsmith reads a copy of master.db and never writes to your library, but back it up "
    "first anyway: in Rekordbox, File > Library > Backup Library. Then re-run with --backed-up."
)


@rekordbox_app.command("import")
def rekordbox_import(
    master_db: Annotated[
        Path | None,
        typer.Option(
            "--master-db",
            dir_okay=False,
            help="Path to master.db (default: Rekordbox's standard location).",
        ),
    ] = None,
    key: Annotated[
        str | None,
        typer.Option(
            "--key",
            envvar=KEY_ENV,
            show_envvar=True,
            help="master.db key. You supply it; Setsmith never fetches one.",
        ),
    ] = None,
    backed_up: Annotated[
        bool,
        typer.Option("--backed-up", help="Confirm you have backed up your Rekordbox library."),
    ] = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Copy master.db, read My Tags and history from the copy, and store them in Setsmith.

    Your live master.db is only copied, never opened. The copy goes into Setsmith's data
    folder (the newest few are kept). A new import replaces the previous one.
    """
    if not has_module("pyrekordbox"):
        err.print("[red]This needs the rekordbox extra:[/red] uv sync --extra rekordbox")
        raise typer.Exit(EXIT_LOAD_ERROR)
    if not backed_up:
        err.print(f"[yellow]{BACKUP_WARNING}[/yellow]")
        raise typer.Exit(EXIT_TRACK_NOT_FOUND)
    source = master_db or default_master_db()
    if source is None:
        raise typer.BadParameter("master.db not found; pass its path", param_hint="--master-db")
    if not key:
        raise typer.BadParameter(
            f"give the master.db key with --key or ${KEY_ENV}", param_hint="--key"
        )
    running = rekordbox_running()
    if running and not as_json:
        err.print(
            "[yellow]Rekordbox is running; the copy may miss its latest changes. "
            "Quit Rekordbox for a clean snapshot.[/yellow]"
        )

    db_path = db or default_db_path()
    copies_dir = db_path.parent / "rekordbox-copies"
    try:
        copy = copy_master_db(source, copies_dir, DEFAULT_CONFIG.rekordbox_db.keep_copies)
        data = read_master_db(copy, key)
    except RekordboxDbError as exc:
        err.print(f"[red]Import failed:[/red] {exc}")
        raise typer.Exit(EXIT_LOAD_ERROR) from exc
    with Store(db_path) as store:
        store.save_rekordbox_data(data, str(source))

    tag_counts = Counter(t for tags in data.my_tags.values() for t in tags)
    if as_json:
        _emit_json(
            {
                "source": str(source),
                "copy": str(copy),
                "rekordbox_running": running,
                "tracks": data.content_count,
                "tracks_tagged": len(data.my_tags),
                "my_tags": dict(tag_counts.most_common()),
                "sessions": len(data.sessions),
            }
        )
        return
    out.print(f"Read a copy of {source} ({data.content_count} tracks).")
    out.print(f"My Tags on {len(data.my_tags)} tracks; {len(data.sessions)} history sessions.")
    if tag_counts:
        top = ", ".join(f"{t} ({c})" for t, c in tag_counts.most_common(10))
        out.print(f"[bold]Top tags[/bold]  {top}", highlight=False)
    out.print(
        "[dim]suggest, build and info now use My Tags (vocal/instrumental tags set the vocal "
        "flag); build --tag filters by them.[/dim]"
    )


@rekordbox_app.command("status")
def rekordbox_status(db: DbOpt = None, as_json: JsonOpt = False) -> None:
    """Show what was imported from master.db, and when."""
    db_path = db or default_db_path()
    info = None
    if db_path.exists():
        with Store(db_path) as store:
            info = store.rekordbox_import_info()
    if as_json:
        _emit_json(info)
        return
    if info is None:
        out.print("Nothing imported yet. See: setsmith rekordbox import --help")
        return
    out.print(
        f"Imported {info['imported_at']} from {info['source']}: {info['tracks_tagged']} tagged "
        f"tracks, {info['sessions']} history sessions.",
        highlight=False,
    )


# ---------------------------------------------------------------- live sets

liveset_app = typer.Typer(
    help="Analyze DJ sets you supply (tracklist, optional recording) and draft style "
    "profiles from them. Nothing is downloaded or scraped; only derived numbers are stored.",
    no_args_is_help=True,
)
app.add_typer(liveset_app, name="liveset")


def _mmss(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _print_liveset(ls: LiveSet, liveset_id: int | None) -> None:
    stats = compute_stats(ls)
    header = f"[bold]{ls.name}[/bold]" + (f"  [dim]#{liveset_id}[/dim]" if liveset_id else "")
    out.print(header)
    source = Path(ls.tracklist_source).name
    audio = (
        f", recording {Path(ls.audio_source).name} ({_mmss(ls.duration_s)})"
        if ls.audio_source
        else ""
    )
    out.print(
        f"[dim]{len(ls.matches)} tracklist lines from {source}{audio}: {stats.matched} matched, "
        f"{stats.unmatched} not in your collection, {stats.unknown} unidentified[/dim]",
        highlight=False,
    )
    for warning in ls.warnings:
        out.print(f"[yellow]warning:[/yellow] {warning}", highlight=False)
    for m in ls.unmatched[: DEFAULT_CONFIG.display.max_warnings_shown]:
        out.print(f"[yellow]not matched:[/yellow] {m.raw}", highlight=False)

    table = Table(
        "#", "From", "To", "Key move", "Tempo", "Overlap", "Cue out", "Cue in", pad_edge=False
    )
    names = {m.position: m.track_display or m.raw for m in ls.matches}
    for t in ls.transitions:
        overlap = f"{t.overlap_bars:g} bars" if t.overlap_bars is not None else "-"
        tempo = f"{t.tempo_change_pct:.1f}%" if t.tempo_change_pct is not None else "-"
        table.add_row(
            str(t.from_position), names[t.from_position], names[t.to_position],
            t.key_move or "-", tempo, overlap, _mmss(t.cue_out_s), _mmss(t.cue_in_s),
        )  # fmt: skip
    if ls.transitions:
        out.print(table)
    if stats.track_bpms:
        out.print(f"[bold]Tempo[/bold]  {min(stats.track_bpms):g}-{max(stats.track_bpms):g} BPM")
    if stats.key_moves:
        moves = ", ".join(
            f"{k} {v}" for k, v in sorted(stats.key_moves.items(), key=lambda kv: -kv[1])
        )
        out.print(f"[bold]Key moves[/bold]  {moves}")
    if stats.transition_bars:
        lengths = ", ".join(
            f"{k} bars: {v}"
            for k, v in sorted(stats.transition_bars.items(), key=lambda kv: int(kv[0]))
        )
        out.print(f"[bold]Transition lengths[/bold]  {lengths}")
    if stats.genre_mix:
        out.print(
            f"[bold]Genres[/bold]  {', '.join(f'{g} {c}' for g, c in stats.genre_mix.items())}"
        )


def _save_profile(draft: dict[str, Any], key: str, force: bool) -> Path:
    try:
        StyleProfile.model_validate(draft)
    except ValueError as exc:
        err.print(f"[red]The drafted profile is not valid:[/red] {exc}")
        raise typer.Exit(EXIT_BUILD_ERROR) from exc
    target = user_styles_dir() / f"{key}.json"
    if target.exists() and not force:
        err.print(f"[red]{target} already exists.[/red] Pass --force to replace it.")
        raise typer.Exit(EXIT_BUILD_ERROR)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


SaveProfileOpt = Annotated[
    str | None,
    typer.Option(
        "--save-profile",
        help="Save the drafted style profile under this name in your styles folder.",
    ),
]


@liveset_app.command("analyze")
def liveset_analyze(
    collection: CollectionArg,
    tracklist: Annotated[
        Path,
        typer.Argument(
            help="Tracklist: pasted text or CSV, one track per line.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    audio: Annotated[
        Path | None,
        typer.Option(
            "--audio",
            exists=True,
            dir_okay=False,
            readable=True,
            help="A recording of the set that you are entitled to analyze (WAV, FLAC, MP3).",
        ),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Name for this set.")] = None,
    align: Annotated[
        bool,
        typer.Option(help="Align your original files to the recording to measure transitions."),
    ] = True,
    no_essentia: Annotated[
        bool, typer.Option("--no-essentia", help="Use librosa for window keys.")
    ] = False,
    save_profile: SaveProfileOpt = None,
    force: Annotated[bool, typer.Option(help="Replace an existing saved profile.")] = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Match a tracklist to your collection and measure the set; draft a style profile.

    With --audio, the recording is read in 30-second blocks (never copied): tempo, key and
    loudness per window, plus where each of your original files plays in it.
    """
    if audio is not None and not has_module("librosa"):
        err.print("[red]Analyzing a recording needs the audio extra:[/red] uv sync --extra audio")
        raise typer.Exit(EXIT_LOAD_ERROR)
    col, _ = _load_analyzed(collection, db, use_analysis=True)
    entries = load_tracklist(tracklist)
    if not entries:
        raise typer.BadParameter("no tracks found in the tracklist", param_hint="TRACKLIST")
    matches = match_tracklist(col, entries)
    set_name = name or tracklist.stem

    def progress(message: str) -> None:
        if not as_json:
            status.update(f"[dim]{message}...[/dim]")

    try:
        with out.status("[dim]analyzing...[/dim]") as status:
            ls = analyze_liveset(
                set_name, str(tracklist), matches, audio=audio, align=align,
                use_essentia=not no_essentia, on_progress=progress,
            )  # fmt: skip
    except AnalysisError as exc:
        err.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(EXIT_BUILD_ERROR) from exc

    with Store(db) as store:
        liveset_id = store.save_liveset(ls.to_dict())
    draft = draft_profile(ls)
    saved = _save_profile(draft, save_profile, force) if save_profile else None

    if as_json:
        _emit_json(
            {
                "id": liveset_id,
                "liveset": ls.to_dict(),
                "stats": compute_stats(ls).to_dict(),
                "draft_profile": draft,
                "saved_profile": str(saved) if saved else None,
            }
        )
        return
    _print_liveset(ls, liveset_id)
    out.print("\n[bold]Draft style profile[/bold] [dim](edit before use)[/dim]")
    out.print_json(json.dumps(draft, ensure_ascii=False))
    if saved:
        out.print(f"Saved to [bold]{saved}[/bold]; use it with --style {saved.stem}.")
    else:
        out.print(
            "[dim]Save it with --save-profile NAME, or later: setsmith liveset profile "
            f"{liveset_id} --save-profile NAME[/dim]"
        )


def _get_liveset(liveset_id: int, db: Path | None) -> LiveSet:
    db_path = db or default_db_path()
    data = None
    if db_path.exists():
        with Store(db_path) as store:
            data = store.get_liveset(liveset_id)
    if data is None:
        err.print(f"[red]No stored live set #{liveset_id}.[/red] See: setsmith liveset list")
        raise typer.Exit(EXIT_TRACK_NOT_FOUND)
    return LiveSet.from_dict(data)


@liveset_app.command("list")
def liveset_list(db: DbOpt = None, as_json: JsonOpt = False) -> None:
    """List analyzed live sets."""
    db_path = db or default_db_path()
    rows: list[tuple[int, str, str]] = []
    if db_path.exists():
        with Store(db_path) as store:
            rows = store.list_livesets()
    if as_json:
        _emit_json([{"id": i, "name": n, "created_at": c} for i, n, c in rows])
        return
    if not rows:
        out.print("No live sets analyzed yet. See: setsmith liveset analyze --help")
        return
    table = Table("#", "Name", "Analyzed", pad_edge=False)
    for i, n, c in rows:
        table.add_row(str(i), n, c[:16].replace("T", " "))
    out.print(table)


@liveset_app.command("show")
def liveset_show(
    liveset_id: Annotated[int, typer.Argument(help="Live set number from 'liveset list'.")],
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Show a stored live set's transitions and statistics."""
    ls = _get_liveset(liveset_id, db)
    if as_json:
        _emit_json({"liveset": ls.to_dict(), "stats": compute_stats(ls).to_dict()})
        return
    _print_liveset(ls, liveset_id)


@liveset_app.command("profile")
def liveset_profile(
    liveset_id: Annotated[int, typer.Argument(help="Live set number from 'liveset list'.")],
    save_profile: SaveProfileOpt = None,
    force: Annotated[bool, typer.Option(help="Replace an existing saved profile.")] = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Draft a style profile from a stored live set."""
    draft = draft_profile(_get_liveset(liveset_id, db))
    saved = _save_profile(draft, save_profile, force) if save_profile else None
    if as_json:
        _emit_json({"draft_profile": draft, "saved_profile": str(saved) if saved else None})
        return
    out.print_json(json.dumps(draft, ensure_ascii=False))
    if saved:
        out.print(f"Saved to [bold]{saved}[/bold]; use it with --style {saved.stem}.")


@liveset_app.command("delete")
def liveset_delete(
    liveset_id: Annotated[int, typer.Argument(help="Live set number from 'liveset list'.")],
    db: DbOpt = None,
) -> None:
    """Delete a stored live set (its derived data only; no audio is ever stored)."""
    with Store(db) as store:
        deleted = store.delete_liveset(liveset_id)
    if not deleted:
        err.print(f"[red]No stored live set #{liveset_id}.[/red]")
        raise typer.Exit(EXIT_TRACK_NOT_FOUND)
    out.print(f"Deleted live set #{liveset_id}.")


# ---------------------------------------------------------------- learning


@app.command("learn")
def learn_command(
    collection: CollectionArg,
    prior: Annotated[
        float,
        typer.Option(
            min=0,
            help="How many transitions it takes for your habits to count as much as the defaults.",
        ),
    ] = DEFAULT_CONFIG.liveset.learn_prior,
    history: Annotated[bool, typer.Option(help="Learn from imported Rekordbox history.")] = True,
    livesets: Annotated[bool, typer.Option(help="Learn from analyzed live sets.")] = True,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Learn your key-move and tempo-change habits and store them for --learned.

    Uses consecutive tracks in Rekordbox history sessions ('setsmith rekordbox import') and
    transitions in analyzed live sets. Learned values blend with the defaults with weight
    n / (n + prior).
    """
    col, _ = _load_analyzed(collection, db, use_analysis=True)
    by_path: dict[str, Track] = {}
    for track in col.tracks.values():
        local = location_to_path(track.location)
        if local:
            by_path[normalize_file_path(local)] = track
    with Store(db) as store:
        sessions = [s.paths for s in store.load_history()] if history else []
        liveset_moves: list[tuple[str | None, float | None]] = []
        if livesets:
            for liveset_id, _, _ in store.list_livesets():
                data = store.get_liveset(liveset_id)
                if data:
                    ls = LiveSet.from_dict(data)
                    liveset_moves += [(t.key_move, t.tempo_change_pct) for t in ls.transitions]
        prefs = learn(history_pairs(sessions, by_path), liveset_moves, prior=prior)
        if prefs.key_count or prefs.tempo_count:
            store.save_learned(prefs.to_dict())

    if as_json:
        _emit_json(
            {
                **prefs.to_dict(),
                "key_table": [
                    dict(zip(("move", "played", "default", "learned"), r, strict=True))
                    for r in move_table(prefs)
                ],
                "tempo_table": [
                    dict(zip(("band", "played", "default", "learned"), r, strict=True))
                    for r in tempo_table(prefs)
                ],
            }
        )
        return
    total = sum(prefs.sources.values())
    if not (prefs.key_count or prefs.tempo_count):
        out.print(
            "Nothing to learn from yet: import history with 'setsmith rekordbox import' or "
            "analyze a set with 'setsmith liveset analyze'."
        )
        return
    out.print(
        f"Learned from {total} transitions ({prefs.sources.get('history', 0)} from history, "
        f"{prefs.sources.get('livesets', 0)} from live sets); prior {prior:g}."
    )
    for title, rows in (("Key moves", move_table(prefs)), ("Tempo change", tempo_table(prefs))):
        table = Table(title, "Played", "Default", "Learned", title=None, pad_edge=False)
        for label, played, default, new in rows:
            table.add_row(label, str(played), f"{default:.2f}", f"{new:.2f}")
        out.print(table)
    out.print("[dim]Use with: setsmith suggest ... --learned / setsmith build ... --learned[/dim]")


# ---------------------------------------------------------------- web UI


@app.command()
def web(
    collection: CollectionArg,
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8765,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the UI in your browser.")
    ] = False,
    db: DbOpt = None,
) -> None:
    """Start the local web UI: build sets on a timeline, suggestions, analyzed live sets.

    Listens on 127.0.0.1 only by default; your collection is read, never written.
    """
    if not has_module("fastapi") or not has_module("uvicorn"):
        err.print("[red]The web UI needs the web extra:[/red] uv sync --extra web")
        raise typer.Exit(EXIT_LOAD_ERROR)
    import uvicorn

    from setsmith.web.app import LOCAL_HOSTS, create_app

    local = host in LOCAL_HOSTS
    if not local:
        err.print(
            f"[yellow]Listening on {host} makes your library browsable by other machines on "
            "the network. Setsmith has no login.[/yellow]"
        )
        if has_module("essentia"):
            err.print(
                "[yellow]Essentia is installed and is AGPL-3.0: serving this app to other "
                "people brings AGPL obligations.[/yellow]"
            )
    try:
        application = create_app(collection, db, extra_hosts=None if local else [host])
    except (RekordboxXMLError, OSError) as exc:
        err.print(f"[red]Could not read {collection}:[/red] {exc}")
        raise typer.Exit(EXIT_LOAD_ERROR) from exc
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    out.print(f"Setsmith UI at [bold]{url}[/bold]  [dim](Ctrl-C to stop)[/dim]")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    uvicorn.run(application, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------- discovery


keys_app = typer.Typer(
    help="API keys for discovery (Last.fm, GetSongBPM). Stored readable by you only.",
    no_args_is_help=True,
)
app.add_typer(keys_app, name="keys")


class Service(StrEnum):
    LASTFM = "lastfm"
    GETSONGBPM = "getsongbpm"


@keys_app.command("set")
def keys_set(
    service: Annotated[Service, typer.Argument(help="lastfm or getsongbpm")],
    key: Annotated[str, typer.Argument(help="Your API key for that service.")],
) -> None:
    """Save an API key (an environment variable with the same purpose takes precedence)."""
    value = key.strip()
    if "PASTE" in value.upper() or "YOUR_" in value.upper():
        raise typer.BadParameter(
            "that's the placeholder text; replace it with the key the service gave you",
            param_hint="KEY",
        )
    if service == Service.LASTFM and not re.fullmatch(r"[0-9a-f]{32}", value):
        err.print(
            "[yellow]Last.fm API keys are 32 characters of 0-9 and a-f; double-check you "
            "copied the 'API key' (not the shared secret).[/yellow]"
        )
    path = set_key(service.value, value)
    out.print(f"Saved your {service.value} key to {path} (readable by you only).")


@keys_app.command("status")
def keys_status(as_json: JsonOpt = False) -> None:
    """Show which keys are configured, without printing them."""
    rows = {name: key_source(name) for name in SERVICES}
    if as_json:
        _emit_json(rows)
        return
    for name, source in rows.items():
        env_name, signup = SERVICES[name]
        state = f"set ({source})" if source else f"missing: get one at {signup}"
        out.print(f"[bold]{name}[/bold]  {state}  [dim](or ${env_name})[/dim]", highlight=False)


@keys_app.command("remove")
def keys_remove(service: Annotated[Service, typer.Argument(help="lastfm or getsongbpm")]) -> None:
    """Delete a saved key from the keys file."""
    removed = remove_key(service.value)
    out.print(f"Removed the saved {service.value} key." if removed else "No saved key to remove.")


def _attribution_line() -> str:
    return "  ·  ".join(f"{text}: {url}" for text, url in DISCOVERY_ATTRIBUTIONS)


@app.command()
def discover(
    collection: CollectionArg,
    track: Annotated[
        str | None,
        typer.Option("--track", "-t", help='Seed track: "Artist - Title", or the title alone.'),
    ] = None,
    track_id: Annotated[
        str | None, typer.Option("--id", help="Seed track by Rekordbox TrackID.")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", min=1, max=50, help="Results to show.")] = 15,
    style: StyleOpt = None,
    learned: LearnedOpt = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Find tracks you don't own that would mix well after a seed track.

    Similar tracks come from Last.fm and BPM/key from GetSongBPM, using your own API keys
    ('setsmith keys set'). Results link to Last.fm, SoundCloud and Beatport so you can
    listen and add them; nothing is downloaded.
    """
    profile = _load_style_opt(style)
    cfg = _scoring_config(learned, db)
    col, _ = _load_analyzed(collection, db, use_analysis=True)
    seed = _resolve_seed(col, track, track_id, as_json)
    try:
        with Store(db) as store:
            lfm, bpm = make_sources(store, cfg)
            with out.status("[dim]discovering...[/dim]") as spinner:
                result = run_discovery(
                    col, seed, lfm, bpm, style=profile, top=top, cfg=cfg,
                    on_progress=None if as_json else lambda m: spinner.update(f"[dim]{m}...[/dim]"),
                )  # fmt: skip
    except DiscoveryUnavailable as exc:
        err.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(EXIT_LOAD_ERROR) from exc
    except (ServiceError, ValueError) as exc:
        err.print(f"[red]Discovery failed:[/red] {exc}")
        raise typer.Exit(EXIT_BUILD_ERROR) from exc

    if as_json:
        _emit_json(result.to_dict())
        return
    out.print(_seed_line(seed))
    out.print(
        f"[dim]Last.fm suggested {result.candidates} tracks (already in your library: "
        f"{result.in_library}); {len(result.items)} shown.[/dim]"
    )
    for warning in result.warnings:
        out.print(f"[yellow]note:[/yellow] {warning}", highlight=False)
    table = Table(pad_edge=False)
    for name, justify in (("#", "right"), ("Track", "left"), ("BPM", "right"), ("Key", "left"),
                          ("Genre", "left"), ("Score", "right"), ("Transition", "left"),
                          ("Found via", "left")):  # fmt: skip
        table.add_column(name, justify=justify, overflow="fold")  # type: ignore[arg-type]
    for i, d in enumerate(result.items, 1):
        t = d.track
        table.add_row(
            str(i), f"{t.artist} - {t.title}", _bpm(t), _key(t), t.genre or "-",
            _score_text(d.score.total),
            f"{d.score.suggested_type.value} {d.score.suggested_length_bars}b", d.via,
        )  # fmt: skip
    out.print(table)
    if result.items:
        first = result.items[0]
        out.print(f"[dim]Listen: {first.links['soundcloud']}  (all links in --json)[/dim]")
    out.print(f"[dim]{_attribution_line()}[/dim]", highlight=False)


# ---------------------------------------------------------------- feedback

feedback_app = typer.Typer(
    help="Log how suggested transitions sounded on real decks, for later weight tuning.",
    no_args_is_help=True,
)
app.add_typer(feedback_app, name="feedback")


class Verdict(StrEnum):
    GOOD = "good"
    BAD = "bad"


@feedback_app.command("log")
def feedback_log(
    collection: CollectionArg,
    verdict: Annotated[Verdict, typer.Argument(help="How the transition sounded.")],
    from_track: Annotated[
        str | None, typer.Option("--from", help='Outgoing track: "Artist - Title".')
    ] = None,
    from_id: Annotated[str | None, typer.Option(help="Outgoing track by TrackID.")] = None,
    to_track: Annotated[
        str | None, typer.Option("--to", help='Incoming track: "Artist - Title".')
    ] = None,
    to_id: Annotated[str | None, typer.Option(help="Incoming track by TrackID.")] = None,
    transition_type: Annotated[
        TransitionType | None,
        typer.Option("--type", help="Transition you played (default: the suggested one)."),
    ] = None,
    bars: Annotated[int | None, typer.Option(min=1, help="Length you played, in bars.")] = None,
    note: Annotated[str, typer.Option(help="Anything worth remembering.")] = "",
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Record one transition as good or bad, with Setsmith's current score for it."""
    col, _ = _load_analyzed(collection, db, use_analysis=True)
    a = _resolve_track(col, from_track, from_id, as_json, flags=("--from", "--from-id"))
    b = _resolve_track(col, to_track, to_id, as_json, flags=("--to", "--to-id"))
    assert a is not None and b is not None
    score = score_transition(a, b)
    kind = transition_type or score.suggested_type
    with Store(db) as store:
        entry_id = store.add_feedback(
            verdict=verdict.value,
            a_id=a.id,
            b_id=b.id,
            a_label=a.display,
            b_label=b.display,
            a_location=a.location,
            b_location=b.location,
            transition_type=kind.value,
            length_bars=bars if bars is not None else score.suggested_length_bars,
            total=score.total,
            score=score.to_dict(),
            note=note,
            collection=str(collection),
        )
    if as_json:
        _emit_json({"id": entry_id, "verdict": verdict.value, "score": score.to_dict()})
        return
    style = "green" if verdict == Verdict.GOOD else "red"
    out.print(
        f"Logged #{entry_id} [{style}]{verdict.value}[/{style}]: {a.display} -> {b.display} "
        f"({kind.value}, Setsmith scored {score.total:.0f})",
        highlight=False,
    )


@feedback_app.command("list")
def feedback_list(
    db: DbOpt = None,
    limit: Annotated[int, typer.Option(min=1, help="Show the most recent N entries.")] = 50,
    as_json: JsonOpt = False,
) -> None:
    """Show logged feedback and how often Setsmith's scores agreed with your ears."""
    with Store(db) as store:
        entries = list(store.iter_feedback())
    if as_json:
        _emit_json([e.to_dict() for e in entries[-limit:]])
        return
    if not entries:
        out.print(
            "No feedback logged yet. Try: setsmith feedback log <xml> good --from ... --to ..."
        )
        return
    table = Table("#", "When", "Verdict", "Transition", "Type", "Score", "Note", pad_edge=False)
    for e in entries[-limit:]:
        style = "green" if e.verdict == Verdict.GOOD else "red"
        table.add_row(
            str(e.id),
            e.created_at[:16].replace("T", " "),
            Text(e.verdict, style=style),
            f"{e.a_label} -> {e.b_label}",
            f"{e.transition_type} {e.length_bars or ''}".strip(),
            f"{e.total:.0f}",
            e.note,
        )
    out.print(table)
    good = [e.total for e in entries if e.verdict == Verdict.GOOD]
    bad = [e.total for e in entries if e.verdict == Verdict.BAD]
    parts = [f"{len(entries)} logged: {len(good)} good, {len(bad)} bad"]
    if good:
        parts.append(f"mean score of good {sum(good) / len(good):.0f}")
    if bad:
        parts.append(f"of bad {sum(bad) / len(bad):.0f}")
    out.print(f"[dim]{', '.join(parts)}[/dim]")


if __name__ == "__main__":
    app()

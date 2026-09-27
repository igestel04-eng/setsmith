"""Setsmith command line interface."""

from __future__ import annotations

import json
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
from setsmith.graph.build import config_key, track_fingerprint
from setsmith.io.rekordbox_xml import (
    ExportError,
    PlaylistSpec,
    RekordboxXMLError,
    export_playlists,
    load_collection,
)
from setsmith.keys.camelot import classify_move, parse_key
from setsmith.model.collection import Collection, NodeType
from setsmith.model.track import Track
from setsmith.scoring.suggest import Suggestion, suggest_next
from setsmith.scoring.transition import TransitionType, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import GeneratedSet, SetGenerationError, SetRequest, filter_pool
from setsmith.sets.generate import generate_set as run_generation
from setsmith.sets.report import render_markdown, set_comments, set_to_dict, summary_line
from setsmith.store import Store, default_db_path

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
) -> tuple[Collection, AnalysisCoverage | None]:
    """Load a collection and apply stored audio analysis, if there is any."""
    col = _load(path)
    if not use_analysis:
        return col, None
    db_path = db or default_db_path()
    if not db_path.exists():  # never create a database just to read from it
        return col, None
    with Store(db_path) as store:
        return col, load_and_apply(col, store)


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
    db: DbOpt = None,
    no_analysis: NoAnalysisOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Suggest the best tracks to play after a seed track, with a score breakdown."""
    col, _ = _load_analyzed(collection, db, not no_analysis)
    seed = _resolve_seed(col, track, track_id, as_json)
    results = suggest_next(col, seed, top=top, energy_target=energy_delta)

    if as_json:
        _emit_json(
            {
                "seed": seed.summary(),
                "energy_target": energy_delta,
                "suggestions": [
                    {"rank": i, "track": s.track.summary(), "score": s.score.to_dict()}
                    for i, s in enumerate(results, 1)
                ],
            }
        )
        return

    out.print(_seed_line(seed))
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
    col, analysis = _load_analyzed(collection, db, not no_analysis)
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
        str,
        typer.Option(
            "--curve",
            "-c",
            help="Energy curve: warm_up, peak_time, closing, journey, or custom points "
            "such as '3,5,8,6' or '0:3,0.6:9,1:5'.",
        ),
    ] = "journey",
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
    try:
        energy_curve = parse_curve(curve)
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
    )

    store = None if no_cache else Store(db)
    cfg_key = config_key(DEFAULT_CONFIG)
    try:
        cache = None
        if store is not None:
            pool, _ = filter_pool(col, request)
            cache = store.load_pair_scores(cfg_key, {track_fingerprint(t) for t in pool})
        generated, graph = run_generation(col, request, cache=cache)
        if store is not None:
            store.save_pair_scores(cfg_key, graph.new_scores)
    except SetGenerationError as exc:
        err.print(f"[red]Could not build a set:[/red] {exc}")
        raise typer.Exit(EXIT_BUILD_ERROR) from exc
    finally:
        if store is not None:
            store.close()

    length = f"{tracks} tracks" if tracks else f"{minutes:g}min"
    set_name = name or f"{energy_curve.name} {length} {datetime.now():%Y-%m-%d %H.%M}"

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

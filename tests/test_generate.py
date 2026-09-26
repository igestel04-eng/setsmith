import statistics

import pytest

from setsmith.graph.build import CompatibilityGraph, config_key, track_fingerprint
from setsmith.model.collection import Collection
from setsmith.model.track import Track
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import (
    SetGenerationError,
    SetRequest,
    _Search,
    _State,
    artist_names,
    generate_set,
)

CURVES = ["warm_up", "peak_time", "closing", "journey"]


def request(curve: str = "journey", **kw: object) -> SetRequest:
    return SetRequest(curve=parse_curve(curve), **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("curve", CURVES)
def test_energy_curve_adherence(collection_pool: Collection, curve: str) -> None:
    gen, _ = generate_set(collection_pool, request(curve, track_count=16))
    assert gen.stats.track_count == 16
    assert gen.stats.curve_mae is not None
    assert gen.stats.curve_mae < 1.5


@pytest.mark.parametrize("curve", CURVES)
def test_no_artist_repeat_within_four(collection_pool: Collection, curve: str) -> None:
    gen, _ = generate_set(collection_pool, request(curve, track_count=20))
    artists = [artist_names(p.track.artist) for p in gen.positions]
    for i, names in enumerate(artists):
        for j in range(max(0, i - 4), i):
            assert not names & artists[j], f"{gen.positions[i].track.artist} repeats at {i}"


def test_bpm_band_respected(collection_pool: Collection) -> None:
    gen, _ = generate_set(collection_pool, request(track_count=12, bpm_min=120, bpm_max=124))
    assert all(120 <= (p.track.bpm or 0) <= 124 for p in gen.positions)
    for p in gen.positions:
        assert all(120 <= (a.track.bpm or 0) <= 124 for a in p.alternates)


def test_no_duplicates_and_transitions_between_every_pair(collection_pool: Collection) -> None:
    gen, _ = generate_set(collection_pool, request(track_count=18))
    ids = gen.track_ids
    assert len(ids) == len(set(ids)) == 18
    assert all(p.transition is not None for p in gen.positions[:-1])
    assert gen.positions[-1].transition is None
    for p, nxt in zip(gen.positions, gen.positions[1:], strict=False):
        assert p.transition is not None
        assert (p.transition.a_id, p.transition.b_id) == (p.track.id, nxt.track.id)


def test_alternates(collection_pool: Collection) -> None:
    gen, _ = generate_set(collection_pool, request("warm_up", track_count=12))
    path = set(gen.track_ids)
    for p in gen.positions:
        assert len(p.alternates) <= 2
        for alt in p.alternates:
            assert alt.track.id not in path
            if alt.track.energy is not None:
                assert abs(alt.track.energy - p.target_energy) <= 2.0
    assert sum(len(p.alternates) for p in gen.positions) >= 12
    assert not set(gen.alternate_ids) & path


def test_start_track_and_excludes(collection_pool: Collection) -> None:
    excluded = {"1", "2", "3"}
    start = next(t.id for t in collection_pool.tracks.values() if t.bpm and t.id not in excluded)
    gen, _ = generate_set(
        collection_pool,
        request(track_count=10, start_id=start, exclude_ids=frozenset(excluded)),
    )
    assert gen.track_ids[0] == start
    assert not gen.positions[0].alternates  # the opener was fixed by the user
    assert not set(gen.track_ids) & excluded


def test_genre_filter(collection_pool: Collection) -> None:
    gen, _ = generate_set(
        collection_pool, request(track_count=8, genres=frozenset({"tech house", "house"}))
    )
    assert all(p.track.genre in ("Tech House", "House") for p in gen.positions)


def test_minutes_estimate(collection_pool: Collection) -> None:
    gen, _ = generate_set(collection_pool, request(minutes=90))
    assert 80 * 60 <= gen.stats.estimated_seconds <= 100 * 60


def test_deterministic(collection_pool: Collection) -> None:
    first, _ = generate_set(collection_pool, request(track_count=12))
    second, _ = generate_set(collection_pool, request(track_count=12))
    assert first.track_ids == second.track_ids


def test_cache_reuse_gives_same_set(collection_pool: Collection) -> None:
    first, graph = generate_set(collection_pool, request(track_count=12))
    assert first.stats.pairs_scored > 0
    second, _ = generate_set(collection_pool, request(track_count=12), cache=graph.new_scores)
    assert second.track_ids == first.track_ids
    assert second.stats.pairs_scored == 0


def test_mean_transition_quality(collection_pool: Collection) -> None:
    gen, _ = generate_set(collection_pool, request(track_count=16))
    assert gen.stats.mean_transition >= 80
    totals = [p.transition.total for p in gen.positions if p.transition]
    assert statistics.fmean(totals) == pytest.approx(gen.stats.mean_transition)


def test_too_small_pool(collection_5: Collection) -> None:
    with pytest.raises(SetGenerationError, match="match the filters"):
        generate_set(collection_5, request(track_count=4, bpm_min=200))
    with pytest.raises(SetGenerationError, match="minutes or a track count"):
        generate_set(collection_5, request())


def test_short_pool_warns(collection_5: Collection) -> None:
    gen, _ = generate_set(collection_5, request(track_count=10))
    assert gen.stats.track_count <= 4  # the fifth track has no BPM
    assert any("only 4 match" in w for w in gen.warnings)
    assert any("without BPM" in w for w in gen.warnings)


def test_unknown_start_track(collection_5: Collection) -> None:
    with pytest.raises(SetGenerationError, match="not in collection"):
        generate_set(collection_5, request(track_count=3, start_id="nope"))
    with pytest.raises(SetGenerationError, match="no BPM"):
        generate_set(collection_5, request(track_count=3, start_id="105"))


@pytest.mark.parametrize(
    ("artist", "names"),
    [
        ("&ME, Rampa & Adam Port", {"me", "rampa", "adam port"}),
        ("Black Coffee feat. Msaki", {"black coffee", "msaki"}),
        ("Artbat x Sevdaliza", {"artbat", "sevdaliza"}),
        ("Malcolm X", {"malcolm x"}),
        ("X", {"x"}),
        ("Florence and the Machine", {"florence and the machine"}),
        ("", set()),
    ],
)
def test_artist_names(artist: str, names: set[str]) -> None:
    assert artist_names(artist) == names


# ---------------------------------------------------------------- penalties


def mk(tid: str, bpm: float = 124.0, **kw: object) -> Track:
    return Track(id=tid, bpm=bpm, camelot="8A", energy=5.0, genre="House", **kw)  # type: ignore[arg-type]


def search_for(tracks: list[Track], targets: list[float]) -> _Search:
    graph = CompatibilityGraph(tracks)
    return _Search(graph, targets, request(track_count=len(targets)), DEFAULT_CONFIG)


def state(
    path: list[str], boosts: tuple[bool, ...] = (), lo: float = 124.0, hi: float = 124.0
) -> _State:
    return _State(tuple(path), 0.0, boosts, 124.0, lo, hi)


def test_penalty_artist_repeat_and_label_run() -> None:
    tracks = [
        mk("a", artist="X", label="L"),
        mk("b", artist="Y", label="L"),
        mk("c", artist="X", label="L"),
    ]
    s = search_for(tracks, [5.0, 5.0, 5.0])
    penalty, _, _ = s.penalties(state(["a", "b"]), s.t["c"], 2, boost=False)
    assert penalty == 15 + 5


def test_penalty_stacked_energy_boosts() -> None:
    tracks = [mk(str(i)) for i in range(4)]
    s = search_for(tracks, [5.0] * 4)
    two_boosts = state(["0", "1", "2"], boosts=(True, True))
    assert s.penalties(two_boosts, s.t["3"], 3, boost=True)[0] == 10
    assert s.penalties(two_boosts, s.t["3"], 3, boost=False)[0] == 0


def test_penalty_tempo_drift_and_reversal() -> None:
    tracks = [mk("a", 120.0), mk("b", 124.0), mk("c", 130.0), mk("d", 118.0)]
    s = search_for(tracks, [4.0, 5.0, 6.0])
    # range 120-124 grows to 120-130 > 8 BPM
    assert s.penalties(state(["a", "b"], lo=120, hi=124), s.t["c"], 2, boost=False)[0] == 20
    # energy rising while tempo drops 6 BPM (and range stays within 8 BPM)
    assert s.penalties(state(["b"], lo=118, hi=124), s.t["d"], 1, boost=False)[0] == 10
    # same drop while the curve falls: no reversal penalty
    falling = search_for(tracks, [6.0, 5.0])
    assert falling.penalties(state(["b"], lo=118, hi=124), falling.t["d"], 1, boost=False)[0] == 0


def test_half_time_tracks_fold_into_range() -> None:
    tracks = [mk("a", 124.0), mk("b", 62.0)]
    s = search_for(tracks, [5.0, 5.0])
    assert s.fold(62.0, 124.0) == 124.0
    assert s.penalties(state(["a"]), s.t["b"], 1, boost=False)[0] == 0


def test_graph_tempo_window() -> None:
    tracks = [mk("a", 124.0), mk("b", 133.0), mk("c", 135.0), mk("d", 62.5), mk("e", None)]  # type: ignore[arg-type]
    g = CompatibilityGraph(tracks)
    assert g.tempo_neighbors("a") == ["b", "d"]  # 135 is 8.1% away; e has no BPM
    assert g.in_window("a", "d") and not g.in_window("a", "c")
    assert g.tempo_neighbors("e") == []


def test_fingerprint_changes_with_tags() -> None:
    a = mk("a")
    before = track_fingerprint(a)
    a.camelot = "9A"
    assert track_fingerprint(a) != before
    assert config_key(DEFAULT_CONFIG) == config_key(DEFAULT_CONFIG.model_copy())


def test_duplicate_entries_of_a_song_play_once(collection_50: Collection) -> None:
    # TrackIDs 1 and 5 are the same song; the set must not contain both.
    gen, _ = generate_set(collection_50, request("warm_up", track_count=8, start_id="1"))
    assert "5" not in gen.track_ids
    assert "5" not in gen.alternate_ids

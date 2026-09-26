import time
from pathlib import Path

import pytest
from synth import random_tracks, write_collection

from setsmith.io.rekordbox_xml import load_collection
from setsmith.model.collection import Collection
from setsmith.scoring.suggest import suggest_next
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import SetRequest, generate_set

N_TRACKS = 20_000


@pytest.fixture(scope="module")
def big_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    xml = tmp_path_factory.mktemp("perf") / "big.xml"
    write_collection(xml, random_tracks(N_TRACKS, seed=1))
    return xml


@pytest.fixture(scope="module")
def big(big_xml: Path) -> Collection:
    return load_collection(big_xml)


@pytest.mark.perf
def test_20k_collection_load_and_suggest(big_xml: Path) -> None:
    start = time.perf_counter()
    collection = load_collection(big_xml)
    load_s = time.perf_counter() - start
    assert len(collection) == N_TRACKS
    assert load_s < 30, f"load took {load_s:.1f}s"

    seed = next(iter(collection.tracks.values()))
    start = time.perf_counter()
    match = collection.find(f"{seed.artist} - {seed.title}")
    assert match.track is not None
    results = suggest_next(collection, match.track, top=10)
    suggest_s = time.perf_counter() - start
    assert len(results) == 10
    assert suggest_s < 2, f"find + suggest took {suggest_s:.2f}s"
    print(f"\n20k tracks: load {load_s:.2f}s, find + suggest {suggest_s:.2f}s")


@pytest.mark.perf
def test_20k_collection_build_90_minutes(big: Collection) -> None:
    # ~4,600-track pool after the BPM filter; no cache.
    start = time.perf_counter()
    gen, _ = generate_set(
        big, SetRequest(curve=parse_curve("journey"), minutes=90, bpm_min=118, bpm_max=128)
    )
    build_s = time.perf_counter() - start
    assert gen.stats.track_count >= 15
    assert build_s < 20, f"cold build took {build_s:.1f}s"
    print(f"\n20k tracks: cold 90-minute build from {gen.stats.pool_size} tracks {build_s:.1f}s")

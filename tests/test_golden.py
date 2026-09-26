from setsmith.model.collection import Collection
from setsmith.scoring.suggest import suggest_next
from setsmith.scoring.transition import TransitionType


def top_ids(collection: Collection, seed_id: str, n: int) -> list[str]:
    seed = collection.tracks[seed_id]
    return [s.track.id for s in suggest_next(collection, seed, top=n)]


def test_fixture_5_top_matches(collection_5: Collection) -> None:
    assert top_ids(collection_5, "101", 2) == ["102", "103"]


def test_fixture_50_loads(collection_50: Collection) -> None:
    assert len(collection_50) == 50


def test_fixture_50_cluster_a(collection_50: Collection) -> None:
    results = suggest_next(collection_50, collection_50.tracks["1"], top=3)
    assert [s.track.id for s in results] == ["2", "3", "4"]
    assert [round(s.score.total, 1) for s in results] == [100.0, 97.0, 89.0]
    assert all(s.score.suggested_type == TransitionType.LONG_BLEND for s in results)


def test_fixture_50_skips_duplicate_of_seed(collection_50: Collection) -> None:
    assert "5" not in top_ids(collection_50, "1", 49)


def test_fixture_50_cluster_b(collection_50: Collection) -> None:
    assert top_ids(collection_50, "6", 2) == ["7", "8"]


def test_exclude_ids(collection_50: Collection) -> None:
    seed = collection_50.tracks["1"]
    results = suggest_next(collection_50, seed, top=2, exclude_ids={"2"})
    assert [s.track.id for s in results] == ["3", "4"]


def test_energy_target_reranks(collection_50: Collection) -> None:
    # Asking to build +2 favours the E7 track over the E6 one.
    seed = collection_50.tracks["1"]
    results = suggest_next(collection_50, seed, top=1, energy_target=2.0)
    assert results[0].track.id == "3"

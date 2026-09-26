from pathlib import Path

import pytest

from setsmith.store import Store, default_db_path


def test_pair_scores_round_trip_and_pool_filter(tmp_path: Path) -> None:
    with Store(tmp_path / "s.db") as store:
        store.save_pair_scores("cfg1", {("a", "b"): (70.5, True), ("b", "c"): (60.0, False)})
        store.save_pair_scores("cfg2", {("a", "b"): (10.0, False)})
        assert store.load_pair_scores("cfg1", {"a", "b"}) == {("a", "b"): (70.5, True)}
        assert store.load_pair_scores("cfg1", {"a", "b", "c"}) == {
            ("a", "b"): (70.5, True),
            ("b", "c"): (60.0, False),
        }
        assert store.load_pair_scores("cfg2", {"a", "b"}) == {("a", "b"): (10.0, False)}
        assert store.count_pair_scores() == 3
        assert store.clear_pair_scores() == 3


def test_feedback(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    with Store(db) as store:
        entry_id = store.add_feedback(
            verdict="good", a_id="1", b_id="2", a_label="A - One", b_label="B - Two",
            a_location="file://a", b_location="file://b", transition_type="long_blend",
            length_bars=32, total=91.5, score={"total": 91.5}, note="lovely",
        )  # fmt: skip
        assert entry_id == 1
    with Store(db) as store:  # persists across connections
        (entry,) = store.iter_feedback()
    assert entry.verdict == "good" and entry.note == "lovely" and entry.score == {"total": 91.5}
    assert entry.to_dict()["from"] == {"id": "1", "track": "A - One"}


def test_feedback_rejects_bad_verdict(tmp_path: Path) -> None:
    import sqlite3

    with Store(tmp_path / "s.db") as store, pytest.raises(sqlite3.IntegrityError):
        store.add_feedback(
            verdict="meh", a_id="1", b_id="2", a_label="", b_label="", a_location="",
            b_location="", transition_type="cut", length_bars=None, total=1.0, score={},
        )  # fmt: skip


def test_default_path_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETSMITH_DB", str(tmp_path / "x.db"))
    assert default_db_path() == tmp_path / "x.db"
    monkeypatch.delenv("SETSMITH_DB")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_db_path() == tmp_path / "setsmith" / "setsmith.db"

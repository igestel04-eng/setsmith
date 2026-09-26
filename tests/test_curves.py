from itertools import pairwise

import pytest

from setsmith.sets.curves import parse_curve


def test_warm_up_rises_and_never_exceeds_six() -> None:
    targets = parse_curve("warm_up").targets(20)
    assert targets[0] == 3.0
    assert targets[-1] == pytest.approx(5.0)
    assert all(b >= a for a, b in pairwise(targets))
    assert max(targets) <= 6.0


def test_peak_time_rises_with_dips() -> None:
    targets = parse_curve("peak_time").targets(16)
    assert targets[0] == 6.0 and targets[-1] == pytest.approx(9.0)
    dips = [i for i in range(1, 16) if targets[i] < targets[i - 1]]
    assert dips == [5, 10]
    for i in dips:
        assert 1.0 <= targets[i - 1] - targets[i] <= 2.0


def test_closing_holds_then_eases() -> None:
    targets = parse_curve("closing").targets(21)
    assert targets[:16] == [8.0] * 16
    assert 5.0 <= targets[-1] <= 6.0


def test_journey_arc() -> None:
    curve = parse_curve("journey")
    assert curve.at(0) == 3.0
    assert curve.at(0.7) == 9.0
    assert curve.at(1) == 5.0
    assert curve.at(0.35) == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("3,5,8,6", [3.0, 5.0, 8.0, 6.0]), ("0:3,0.5:9,1:5", [3.0, 6.0, 9.0, 7.0, 5.0])],
)
def test_custom_curves(spec: str, expected: list[float]) -> None:
    curve = parse_curve(spec)
    assert curve.name == "custom"
    assert curve.targets(len(expected)) == pytest.approx(expected)


@pytest.mark.parametrize("spec", ["sunrise", "5", "0:3,0.5:x", "0:3,1:12", "0.2:3,1:5", "1:5,0:3"])
def test_bad_curves(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_curve(spec)


def test_short_sets() -> None:
    assert parse_curve("journey").targets(0) == []
    assert parse_curve("journey").targets(1) == [3.0]

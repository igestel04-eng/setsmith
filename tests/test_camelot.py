import pytest

from setsmith.keys.camelot import (
    CAMELOT_TABLE,
    CamelotKey,
    KeyMove,
    all_keys,
    camelot_step,
    classify_move,
    normalize_key,
    parse_key,
)

# Every key from the spec table, written three ways: Camelot, long classic, Rekordbox short.
SPEC_KEYS = [
    ("1A", "Ab minor", "Abm"),
    ("2A", "Eb minor", "Ebm"),
    ("3A", "Bb minor", "Bbm"),
    ("4A", "F minor", "Fm"),
    ("5A", "C minor", "Cm"),
    ("6A", "G minor", "Gm"),
    ("7A", "D minor", "Dm"),
    ("8A", "A minor", "Am"),
    ("9A", "E minor", "Em"),
    ("10A", "B minor", "Bm"),
    ("11A", "F# minor", "F#m"),
    ("12A", "Db minor", "Dbm"),
    ("1B", "B major", "B"),
    ("2B", "F# major", "F#"),
    ("3B", "Db major", "Db"),
    ("4B", "Ab major", "Ab"),
    ("5B", "Eb major", "Eb"),
    ("6B", "Bb major", "Bb"),
    ("7B", "F major", "F"),
    ("8B", "C major", "C"),
    ("9B", "G major", "G"),
    ("10B", "D major", "D"),
    ("11B", "A major", "A"),
    ("12B", "E major", "E"),
]

# Open Key = ((Camelot - 8) mod 12) + 1
OPEN_KEY = {
    "1A": "6m", "2A": "7m", "3A": "8m", "4A": "9m", "5A": "10m", "6A": "11m",
    "7A": "12m", "8A": "1m", "9A": "2m", "10A": "3m", "11A": "4m", "12A": "5m",
    "1B": "6d", "2B": "7d", "3B": "8d", "4B": "9d", "5B": "10d", "6B": "11d",
    "7B": "12d", "8B": "1d", "9B": "2d", "10B": "3d", "11B": "4d", "12B": "5d",
}  # fmt: skip


def test_spec_table_covers_all_24_keys() -> None:
    assert len(SPEC_KEYS) == 24
    assert len({c for c, _, _ in SPEC_KEYS}) == 24
    assert [str(k) for k in all_keys()] == [c for c, _, _ in SPEC_KEYS]


@pytest.mark.parametrize(("camelot", "long_name", "short_name"), SPEC_KEYS)
def test_all_notations_resolve_to_same_key(camelot: str, long_name: str, short_name: str) -> None:
    key = parse_key(camelot)
    assert key is not None
    assert str(key) == camelot
    assert key.name == long_name
    assert key.short_name == short_name
    assert parse_key(long_name) == key
    assert parse_key(short_name) == key
    assert parse_key(OPEN_KEY[camelot]) == key
    assert key.open_key == OPEN_KEY[camelot]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # sharps and flats for every black key, minor and major
        ("G#m", "1A"), ("Abm", "1A"),
        ("D#m", "2A"), ("Ebm", "2A"),
        ("A#m", "3A"), ("Bbm", "3A"),
        ("Gbm", "11A"), ("F#m", "11A"),
        ("C#m", "12A"), ("Dbm", "12A"),
        ("Gb", "2B"), ("F#", "2B"),
        ("C#", "3B"), ("Db", "3B"),
        ("G#", "4B"), ("Ab", "4B"),
        ("D#", "5B"), ("Eb", "5B"),
        ("A#", "6B"), ("Bb", "6B"),
        # white-key enharmonics
        ("Cb", "1B"), ("E#m", "4A"), ("Fb", "12B"), ("B#", "8B"),
        # unicode accidentals
        ("G♯m", "1A"), ("B♭", "6B"),
    ],
)  # fmt: skip
def test_enharmonics(raw: str, expected: str) -> None:
    assert normalize_key(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8a", "8A"),
        (" 8A ", "8A"),
        ("12b", "12B"),
        ("8 A", "8A"),
        ("am", "8A"),
        ("A min", "8A"),
        ("A Minor", "8A"),
        ("C maj", "8B"),
        ("Cmaj", "8B"),
        ("C Major", "8B"),
        ("bbm", "3A"),
        ("1M", "8A"),
        ("1D", "8B"),
        ("F#m ", "11A"),
    ],
)
def test_formatting_variants(raw: str, expected: str) -> None:
    assert normalize_key(raw) == expected


@pytest.mark.parametrize(
    "raw", [None, "", "   ", "13A", "0A", "13m", "0d", "H", "Xm", "8C", "A##", "BBm", "Am7", "?"]
)
def test_unrecognized_returns_none(raw: str | None) -> None:
    assert parse_key(raw) is None
    assert normalize_key(raw) is None


def test_camelot_key_validates() -> None:
    with pytest.raises(ValueError):
        CamelotKey(0, "A")
    with pytest.raises(ValueError):
        CamelotKey(13, "B")


def test_camelot_table_matches_circle_of_fifths() -> None:
    # Each step on the wheel is a perfect fifth (7 semitones).
    pitch = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8,
             "A": 9, "Bb": 10, "B": 11}  # fmt: skip
    for n in range(1, 12):
        for col in (0, 1):
            here = pitch[CAMELOT_TABLE[n][col].split()[0]]
            nxt = pitch[CAMELOT_TABLE[n + 1][col].split()[0]]
            assert (nxt - here) % 12 == 7


def k(text: str) -> CamelotKey:
    key = parse_key(text)
    assert key is not None
    return key


@pytest.mark.parametrize(
    ("a", "b", "move"),
    [
        ("8A", "8A", KeyMove.SAME),
        ("8A", "9A", KeyMove.ADJACENT),
        ("8A", "7A", KeyMove.ADJACENT),
        ("12A", "1A", KeyMove.ADJACENT),
        ("1B", "12B", KeyMove.ADJACENT),
        ("8A", "8B", KeyMove.RELATIVE),
        ("8B", "8A", KeyMove.RELATIVE),
        ("8A", "10A", KeyMove.ENERGY_BOOST),
        ("11B", "1B", KeyMove.ENERGY_BOOST),
        ("8A", "6A", KeyMove.CLASH),  # -2 is not an energy boost
        ("8A", "9B", KeyMove.DIAGONAL),
        ("8A", "7B", KeyMove.DIAGONAL),
        ("8B", "9A", KeyMove.DIAGONAL),
        ("8A", "3A", KeyMove.SEMITONE_LIFT),  # A minor -> Bb minor
        ("8B", "3B", KeyMove.SEMITONE_LIFT),  # C major -> Db major
        ("8A", "2A", KeyMove.CLASH),
        ("8A", "3B", KeyMove.CLASH),
        ("8A", "2B", KeyMove.CLASH),
    ],
)
def test_classify_move(a: str, b: str, move: KeyMove) -> None:
    assert classify_move(k(a), k(b)) == move


def test_semitone_lift_is_one_semitone_up() -> None:
    # 8A (A minor) +7 -> 3A (Bb minor): A to Bb is one semitone.
    assert k("Am").number + 7 - 12 == k("Bbm").number
    assert classify_move(k("Am"), k("Bbm")) == KeyMove.SEMITONE_LIFT


def test_camelot_step_is_signed_and_wrapped() -> None:
    assert camelot_step(k("8A"), k("9A")) == 1
    assert camelot_step(k("8A"), k("7A")) == -1
    assert camelot_step(k("12A"), k("1A")) == 1
    assert camelot_step(k("1A"), k("12A")) == -1
    assert camelot_step(k("8A"), k("3A")) == -5
    assert camelot_step(k("8A"), k("2A")) == 6

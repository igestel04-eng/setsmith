import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from setsmith.cli import app
from setsmith.graph.build import config_key
from setsmith.keys.camelot import normalize_key
from setsmith.model.collection import Collection
from setsmith.model.track import Track, derive_vocal
from setsmith.scoring.transition import TransitionType, score_transition
from setsmith.scoring.weights import DEFAULT_CONFIG
from setsmith.sets.curves import parse_curve
from setsmith.sets.generate import GeneratedSet, SetRequest, generate_set
from setsmith.sets.report import render_markdown
from setsmith.styles.profile import (
    StyleError,
    StyleProfile,
    list_styles,
    load_style,
    load_style_file,
)

ARTISTS = ["brunello", "franky_rizardo", "keinemusik"]
GENRES = [
    "afro_house",
    "deep_house",
    "house",
    "melodic_house_techno",
    "organic_house",
    "tech_house",
]
BUILTINS = sorted(ARTISTS + GENRES)
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_styles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    styles = tmp_path / "styles"
    monkeypatch.setenv("SETSMITH_STYLES_DIR", str(styles))
    return styles


def profile_data(**overrides: Any) -> dict[str, Any]:
    data = json.loads(
        (Path(__file__).parents[1] / "setsmith/styles/profiles/brunello.json").read_text()
    )
    data.update(overrides)
    assert isinstance(data, dict)
    return data


def write_profile(path: Path, **overrides: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile_data(**overrides)), encoding="utf-8")
    return path


def mk(tid: str = "x", key: str | None = "8A", **kw: Any) -> Track:
    defaults: dict[str, Any] = {"bpm": 124.0, "energy": 6.0, "genre": "Afro House"}
    defaults.update(kw)
    return Track(id=tid, key_raw=key, camelot=normalize_key(key), **defaults)


# ---------------------------------------------------------------- loading and validation


def test_builtin_profiles_load_and_say_inspired_by() -> None:
    entries = list_styles()
    assert [e.key for e in entries] == BUILTINS
    for entry in entries:
        profile = load_style(entry.key)
        assert entry.builtin
        # Artist profiles say "inspired by"; genre profiles are named after the genre.
        assert profile.kind == ("artist" if entry.key in ARTISTS else "genre")
        assert profile.name.startswith("Inspired by ") == (profile.kind == "artist")
        if profile.kind == "genre":  # its main genre is one Setsmith knows
            main = max(profile.genre_weights, key=profile.genre_weights.__getitem__)
            assert main in {g for grp in DEFAULT_CONFIG.genre.neighbor_groups for g in grp}


@pytest.mark.parametrize("name", ["Franky Rizardo", "franky-rizardo", "FRANKY_RIZARDO"])
def test_name_normalization(name: str) -> None:
    assert load_style(name).name == "Inspired by Franky Rizardo"


def test_unknown_style_lists_available() -> None:
    with pytest.raises(StyleError, match="available: afro_house, brunello, deep_house"):
        load_style("daft")


def test_user_profile_overrides_builtin_and_paths_work(
    isolated_styles: Path, tmp_path: Path
) -> None:
    write_profile(isolated_styles / "brunello.json", name="My Brunello edit")
    assert load_style("brunello").name == "My Brunello edit"
    (entry,) = [e for e in list_styles() if e.key == "brunello"]
    assert not entry.builtin
    custom = write_profile(tmp_path / "elsewhere" / "mine.json", name="Mine")
    assert load_style(str(custom)).name == "Mine"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bpm_band": [130, 124]}, "bpm_band"),
        ({"transition_type_mix": {"long_blend": 0.9, "cut": 0.3}}, "sum to 1"),
        ({"transition_length_bars": {"16": 0.5, "32": 0.4}}, "sum to 1"),
        ({"allowed_key_moves": {"same": 1.0, "clash": 0.5}}, "clash"),
        ({"genre_weights": {"house": 1.5}}, "between 0 and 1"),
        ({"vocal_density": 2}, "vocal_density"),
        ({"surprise": True}, "surprise"),
        ({"transition_type_mix": {"scratch": 1.0}}, "transition_type_mix"),
    ],
)
def test_invalid_profiles(tmp_path: Path, overrides: dict[str, Any], message: str) -> None:
    path = write_profile(tmp_path / "bad.json", **overrides)
    with pytest.raises(StyleError, match=message):
        load_style_file(path)


def test_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(StyleError, match="not valid JSON"):
        load_style_file(path)


# ---------------------------------------------------------------- profile helpers


def test_genre_weight() -> None:
    k = load_style("keinemusik")
    assert k.genre_weight("Afro-House") == 1.0
    assert k.genre_weight("Melodic House") == 0.9  # alias of melodic house & techno
    assert k.genre_weight("Trance") == DEFAULT_CONFIG.style.unlisted_genre_weight
    assert k.genre_weight("") == DEFAULT_CONFIG.style.unknown_genre_weight
    assert k.genre_weight("Techno, Afro House") == 1.0  # best of several genres


def test_style_fit_parts_and_bpm_falloff() -> None:
    k = load_style("keinemusik")  # band 117-124, minor 0.8
    fit = k.style_fit(mk(bpm=121.0, key="8A", genre="Afro House", vocal=True))
    assert fit.parts == {"bpm": 1.0, "genre": 1.0, "key_mode": 0.8, "vocal": 0.5}
    assert fit.score == pytest.approx(0.35 + 0.35 + 0.2 * 0.8 + 0.1 * 0.5)
    assert k.bpm_fit(126.0) == pytest.approx(0.5)  # 2 BPM outside, 4 BPM falloff
    assert k.bpm_fit(129.0) == 0.0
    assert k.bpm_fit(None) is None
    unknown = k.style_fit(Track(id="u"))
    assert unknown.score == 0.5 and unknown.parts == {}


def test_long_blend_preference_and_curves() -> None:
    assert load_style("keinemusik").prefers_long_blends()
    assert not load_style("franky_rizardo").prefers_long_blends()
    assert load_style("franky_rizardo").curve().name == "peak_time"
    custom = StyleProfile.model_validate(profile_data(energy_curve=[[0, 4], [0.5, 8], [1, 6]]))
    assert custom.curve().targets(3) == pytest.approx([4.0, 8.0, 6.0])


def test_profile_mode_curve_parse_matches_template() -> None:
    assert load_style("brunello").curve().targets(10) == parse_curve("journey").targets(10)


# ---------------------------------------------------------------- scoring with a style


def test_key_moves_and_genre_blend_into_scores() -> None:
    k = load_style("keinemusik")
    boost = score_transition(mk("a", key="8A"), mk("b", key="10A"), style=k)
    assert boost.components["harmonic"].score == pytest.approx((0.65 + 0.5) / 2)
    assert "style allows 0.5" in boost.components["harmonic"].reason
    tech = score_transition(mk("a"), mk("b", genre="Tech House"), style=k)
    # afro -> tech house: different genres (0.3), style weight for tech house unlisted (0.2)
    assert tech.components["genre_style"].score == pytest.approx((0.3 + 0.2) / 2)


def test_without_style_scores_are_unchanged() -> None:
    plain = score_transition(mk("a", key="8A"), mk("b", key="10A"))
    assert plain.components["harmonic"].score == pytest.approx(0.65)
    assert "style" not in plain.components["harmonic"].reason
    assert score_transition(mk("a"), mk("b")).suggested_length_bars == 32


def test_long_blend_length_follows_style() -> None:
    assert (
        score_transition(mk("a"), mk("b"), style=load_style("keinemusik")).suggested_length_bars
        == 64
    )
    assert (
        score_transition(mk("a"), mk("b"), style=load_style("brunello")).suggested_length_bars == 32
    )


def test_cut_or_echo_follows_style_mix() -> None:
    clash = (mk("a", key="8A"), mk("b", key="2A"))
    assert score_transition(*clash, style=load_style("franky_rizardo")).suggested_type == (
        TransitionType.CUT
    )
    echo_lover = StyleProfile.model_validate(
        profile_data(transition_type_mix={"long_blend": 0.5, "cut": 0.05, "echo_out": 0.45})
    )
    result = score_transition(*clash, style=echo_lover)
    assert result.suggested_type == TransitionType.ECHO_OUT
    assert "style prefers echo outs" in result.type_reason


def test_cache_key_depends_on_style() -> None:
    keys = {config_key(DEFAULT_CONFIG, load_style(n)) for n in BUILTINS}
    keys.add(config_key(DEFAULT_CONFIG))
    assert len(keys) == len(BUILTINS) + 1


# ---------------------------------------------------------------- set generation


@pytest.mark.parametrize("name", BUILTINS)
def test_styled_sets_stay_in_band(collection_pool: Collection, name: str) -> None:
    style = load_style(name)
    gen, _ = generate_set(
        collection_pool, SetRequest(curve=style.curve(), track_count=10, style=style)
    )
    lo, hi = style.bpm_band
    assert all(lo <= (p.track.bpm or 0) <= hi for p in gen.positions)
    assert gen.stats.style == style.name
    # The fixture spans 118-128 BPM: a band that barely overlaps it (Organic House,
    # 108-120) leaves too few tracks to follow the curve closely.
    if min(hi, 128) - max(lo, 118) >= 4:
        assert gen.stats.curve_mae is not None and gen.stats.curve_mae < 1.5
    assert sum(gen.stats.transition_mix.values()) == pytest.approx(1.0)


def test_explicit_bpm_limits_override_band(collection_pool: Collection) -> None:
    style = load_style("keinemusik")  # 117-124; each explicit limit replaces its side
    request = SetRequest(
        curve=style.curve(), track_count=6, style=style, bpm_min=125.0, bpm_max=128.0
    )
    gen, _ = generate_set(collection_pool, request)
    assert all(125 <= (p.track.bpm or 0) <= 128 for p in gen.positions)
    assert SetRequest(curve=style.curve(), style=style, bpm_max=130.0).bpm_range == (117, 130.0)


def test_style_raises_fit_over_unstyled_set(collection_pool: Collection) -> None:
    style = load_style("franky_rizardo")
    lo, hi = style.bpm_band
    styled, _ = generate_set(
        collection_pool, SetRequest(curve=style.curve(), track_count=12, style=style)
    )
    plain, _ = generate_set(
        collection_pool, SetRequest(curve=style.curve(), track_count=12, bpm_min=lo, bpm_max=hi)
    )

    def fit(gen: GeneratedSet) -> float:
        return sum(style.style_fit(p.track).score for p in gen.positions)

    assert fit(styled) > fit(plain)


def test_report_mentions_style(collection_pool: Collection) -> None:
    style = load_style("brunello")
    gen, _ = generate_set(
        collection_pool, SetRequest(curve=style.curve(), track_count=5, style=style)
    )
    text = render_markdown(gen, "Test")
    assert "Style Inspired by Brunello" in text and "no endorsement" in text


# ---------------------------------------------------------------- vocal tags


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("Lanterns", "Vocal Mix", ""), True),
        (("Lanterns (Dub)", "", ""), False),
        (("Lanterns", "", "big vox at the drop"), True),
        (("Lanterns (Instrumental)", "", ""), False),
        (("Lanterns", "", ""), None),
        (("Lanterns (Dub)", "", "vocal chops"), None),  # contradictory: unknown
        (("Dubai Nights", "", ""), None),  # whole words only
    ],
)
def test_derive_vocal(texts: tuple[str, str, str], expected: bool | None) -> None:
    assert derive_vocal(*texts) is expected


def test_parser_sets_vocal(collection_5: Collection) -> None:
    assert collection_5.tracks["101"].vocal is True  # "warm vocal chant"
    assert collection_5.tracks["102"].vocal is None


# ---------------------------------------------------------------- CLI


def test_cli_styles_list_show_copy(isolated_styles: Path) -> None:
    listing = json.loads(runner.invoke(app, ["styles", "list", "--json"]).output)
    assert [e["key"] for e in listing] == BUILTINS

    shown = runner.invoke(app, ["styles", "show", "keinemusik"])
    assert shown.exit_code == 0 and "117-124" in shown.output and "not endorsed" in shown.output
    data = json.loads(runner.invoke(app, ["styles", "show", "keinemusik", "--json"]).output)
    assert data["allowed_key_moves"]["energy_boost"] == 0.5

    copied = runner.invoke(app, ["styles", "copy", "keinemusik", "my_km"])
    assert copied.exit_code == 0
    assert (isolated_styles / "my_km.json").exists()
    assert runner.invoke(app, ["styles", "copy", "keinemusik", "my_km"]).exit_code == 1
    assert load_style("my_km").name == "Inspired by Keinemusik"


def test_cli_list_reports_broken_profiles(isolated_styles: Path) -> None:
    write_profile(isolated_styles / "broken.json", bpm_band=[1, 0])
    listing = json.loads(runner.invoke(app, ["styles", "list", "--json"]).output)
    (broken,) = [e for e in listing if e["key"] == "broken"]
    assert broken["name"] is None and "bpm_band" in broken["error"]


def test_cli_build_and_suggest_with_style(fixture_pool: Path) -> None:
    built = runner.invoke(
        app, ["build", str(fixture_pool), "--tracks", "8", "--style", "keinemusik", "--json"]
    )
    assert built.exit_code == 0, built.output
    data = json.loads(built.output)
    assert data["curve"] == "journey" and data["stats"]["style"] == "Inspired by Keinemusik"
    assert all(117 <= p["track"]["bpm"] <= 124 for p in data["positions"])
    assert data["stats"]["style_transition_mix"]["long_blend"] == 0.7

    seed = data["positions"][0]["track"]["id"]
    suggested = runner.invoke(
        app, ["suggest", str(fixture_pool), "--id", seed, "--style", "keinemusik", "--json"]
    )
    rows = json.loads(suggested.output)["suggestions"]
    assert all(0 <= r["style_fit"]["score"] <= 1 for r in rows)

    table = runner.invoke(app, ["suggest", str(fixture_pool), "--id", seed, "-s", "brunello", "-x"])
    assert "Style" in table.output and "style fit" in table.output


def test_cli_bad_style(fixture_5: Path) -> None:
    result = runner.invoke(app, ["build", str(fixture_5), "--style", "nope"])
    assert result.exit_code == 2

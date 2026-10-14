import dataclasses
import itertools
import warnings

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

import crisispaths as cp
from crisispaths import (
    ScenarioWarning,
    compose,
    describe_axes,
    enumerate_scenarios,
    grid_size,
    list_families,
    load_family,
)
from crisispaths.spec import SpecError, dump_spec, load_spec

US_GRID = 3 * 3 * 3 * 2 * 3 * 3
AI_GRID = 3 * 3 * 4 * 2 * 2


def quiet():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore", ScenarioWarning)
    return ctx


# --- families and grids -----------------------------------------------------


def test_list_families():
    assert list_families() == ["ai_bubble", "us_bankruptcy"]


def test_grid_sizes():
    assert grid_size("us_bankruptcy") == 1
    full_us = {a: list(v["options"]) for a, v in describe_axes("us_bankruptcy")["axes"].items()}
    full_ai = {a: list(v["options"]) for a, v in describe_axes("ai_bubble")["axes"].items()}
    assert grid_size("us_bankruptcy", **full_us) == US_GRID == 486
    assert grid_size("ai_bubble", **full_ai) == AI_GRID == 144
    assert grid_size("us_bankruptcy", usd=["debasement", "haven"], depth="mild") == 2


@pytest.mark.parametrize("family, size", [("us_bankruptcy", US_GRID), ("ai_bubble", AI_GRID)])
def test_full_grid_validates_and_roundtrips(family, size):
    axes = {a: list(v["options"]) for a, v in describe_axes(family)["axes"].items()}
    ctx = quiet()
    try:
        out = list(enumerate_scenarios(family, dedupe=False, **axes))
    finally:
        ctx.__exit__(None, None, None)
    assert len(out) == size == grid_size(family, **axes)
    assert len({spec.key for _, spec in out}) == size
    for choices, spec in out:
        assert spec.metadata["axis_choices"] == choices
        again = load_spec(yaml.safe_load(dump_spec(spec)))
        assert again == spec
        assert again.content_hash == spec.content_hash
        # every judgment parameter documents itself
        for a in spec.anchors.values():
            if a.calibrated is False:
                assert a.rationale, (spec.key, a)
        if spec.aftermath.calibrated is False:
            assert spec.aftermath.rationale


def test_enumerate_scalar_fixed_list_swept_and_order():
    out = list(
        enumerate_scenarios("us_bankruptcy", usd=["debasement", "haven"], depth=["mild", "base"])
    )
    assert [c["usd"] for c, _ in out] == ["debasement", "debasement", "haven", "haven"]
    assert [c["depth"] for c, _ in out] == ["mild", "base", "mild", "base"]
    assert all(c["aftermath"] == "prolonged_bear" for c, _ in out)
    assert grid_size("us_bankruptcy", usd=["debasement", "haven"], depth=["mild", "base"]) == 4


def test_enumerate_errors():
    with pytest.raises(SpecError, match="valid options"):
        list(enumerate_scenarios("us_bankruptcy", usd=["debasement", "nope"]))
    with pytest.raises(SpecError, match="empty"):
        grid_size("us_bankruptcy", usd=[])


# --- describe / errors ------------------------------------------------------


def test_describe_axes_structure():
    d = describe_axes("us_bankruptcy")
    assert d["family"] == "us_bankruptcy"
    assert d["axis_order"] == [
        "complexity", "usd", "aftermath", "rates_credit", "depth", "crash_speed",
    ]
    assert d["axes"]["usd"]["default"] == "neutral"
    assert set(d["axes"]["usd"]["options"]) == {"debasement", "haven", "neutral"}
    assert d["axes"]["usd"]["options"]["debasement"]["summary"]
    assert d["axes"]["rates_credit"]["options"]["shock"]["requires"] == {"complexity": ["full"]}
    ai = describe_axes("ai_bubble")
    assert ai["axis_order"] == [
        "complexity", "crash_source", "aftermath", "tech_spread", "depth",
    ]
    assert set(ai["axes"]["aftermath"]["options"]) == {
        "analog", "prolonged_bear", "v_recovery", "stagnation",
    }


def test_unknown_axis_and_option_list_valid_ones():
    with pytest.raises(SpecError) as e:
        compose("us_bankruptcy", colour="red")
    assert "colour" in str(e.value) and "crash_speed" in str(e.value)
    with pytest.raises(SpecError) as e:
        compose("us_bankruptcy", usd="moon")
    assert "moon" in str(e.value) and "debasement" in str(e.value) and "neutral" in str(e.value)
    with pytest.raises(SpecError, match="unknown family"):
        compose("nope")


# --- independence of axes (the modelling fix) ---------------------------------


@pytest.mark.parametrize("usd", ["debasement", "haven", "neutral"])
@pytest.mark.parametrize("aftermath", ["prolonged_bear", "v_recovery", "stagnation"])
def test_usd_and_aftermath_are_independent(usd, aftermath):
    s = compose("us_bankruptcy", complexity="full", usd=usd, aftermath=aftermath)
    assert s.aftermath.mode == aftermath
    if usd == "neutral":
        assert "USD" not in s.anchors
    else:
        assert (s.anchors["USD"].params["low"] > 0) == (usd == "haven")
    base = compose("us_bankruptcy", complexity="full", usd=usd)
    assert s.anchors == base.anchors  # aftermath never touches anchors


def test_us_axis_values():
    s = compose("us_bankruptcy", complexity="full", usd="debasement", aftermath="v_recovery",
                rates_credit="shock", depth="severe", crash_speed="slow")
    assert set(s.anchors) == {"EQ", "USD", "HAVEN", "RATES", "CREDIT", "VOL"}
    assert s.anchors["EQ"].params["median"] == 0.55
    assert (s.anchors["USD"].params["low"], s.anchors["USD"].params["high"]) == (-0.20, -0.10)
    h = s.anchors["HAVEN"].params
    assert (h["post_drift_low"], h["post_drift_high"]) == (0.15, 0.30)
    assert (s.anchors["RATES"].params["low"], s.anchors["RATES"].params["high"]) == (100, 250)
    assert (s.anchors["CREDIT"].params["low"], s.anchors["CREDIT"].params["high"]) == (150, 350)
    assert s.crash_length == "6M"
    assert s.aftermath.recovery_fraction == (0.5, 0.8)
    assert s.param_meta["crash_length"]["calibrated"] is False
    base = compose("us_bankruptcy", depth="base").anchors["EQ"].params
    assert (base["median"], base["sigma"], base["cap"]) == (0.40, 0.276, 0.95)
    assert compose("us_bankruptcy", depth="mild").anchors["EQ"].params["median"] == 0.25
    assert compose("us_bankruptcy", crash_speed="fast").crash_length == "1M"
    assert compose("us_bankruptcy", crash_speed="base").crash_length == "3M"
    assert compose("us_bankruptcy", aftermath="prolonged_bear").aftermath.bear_annual_drift == -0.2
    hv = compose("us_bankruptcy", usd="haven").anchors["HAVEN"].params
    assert (hv["post_drift_low"], hv["post_drift_high"]) == (-0.05, 0.0)


def test_complexity_levels():
    kw = {"usd": "debasement", "rates_credit": "shock"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        assert set(compose("us_bankruptcy", complexity="minimal", **kw).anchors) == {"EQ"}
        assert set(compose("us_bankruptcy", complexity="standard", **kw).anchors) == {
            "EQ", "USD", "HAVEN",
        }
    full = compose("us_bankruptcy", complexity="full", **kw)
    assert set(full.anchors) == {"EQ", "USD", "HAVEN", "RATES", "CREDIT", "VOL"}
    hist = compose(
        "us_bankruptcy", complexity="full", usd="haven", rates_credit="history_consistent"
    )
    assert set(hist.anchors) == {"EQ", "USD", "HAVEN", "VOL"}


# --- ai_bubble ---------------------------------------------------------------


def test_ai_bubble_axes():
    s = compose("ai_bubble", complexity="standard", crash_source="analog_exact",
                aftermath="analog")
    assert set(s.anchors) == {"EQ", "TECH"}
    assert all(a.cum_dist == "analog" for a in s.anchors.values())
    assert s.analog_mode == "exact" and s.label == "main_case"
    assert "2000-2002 replay, which is itself a 31-month bear: IXIC -78%, SPX -49%" in (
        s.description
    )
    assert s.aftermath.mode == "analog"
    m = compose("ai_bubble", complexity="minimal")
    assert set(m.anchors) == {"EQ"}
    f = compose("ai_bubble", complexity="full")
    assert set(f.anchors) == {"EQ", "TECH"}
    assert f.conditional_factors == ("USD", "RATES", "VOL")
    b = compose("ai_bubble", crash_source="analog_bootstrap")
    assert b.analog_mode == "block_bootstrap"
    p = compose("ai_bubble", crash_source="parametric", aftermath="prolonged_bear")
    assert p.anchors["EQ"].cum_dist == "lognormal_loss"
    assert p.anchors["EQ"].params["median"] == 0.49
    assert p.anchors["EQ"].calibrated is False and p.anchors["EQ"].rationale
    assert p.anchors["TECH"].cum_dist == "lognormal_loss"
    assert p.analog_window is None
    v = compose("ai_bubble", aftermath="v_recovery")
    assert v.label == "COUNTERFACTUAL" and v.aftermath.mode == "v_recovery"
    assert compose("ai_bubble", aftermath="prolonged_bear").label == "main_case"
    t = compose("ai_bubble", tech_spread="amplified_1_5x")
    assert t.anchors["TECH"].params["spread_scale"] == 1.5
    assert t.anchors["TECH"].calibrated is False
    assert "spread_scale" not in compose("ai_bubble").anchors["TECH"].params
    sev = compose("ai_bubble", depth="severe")
    assert sev.anchors["EQ"].params["loss_scale"] > 1
    assert sev.anchors["TECH"].params["loss_scale"] > 1
    assert sev.anchors["EQ"].calibrated is False and sev.anchors["EQ"].rationale
    # replay rationale is kept next to the judgment note (rationales accumulate)
    assert "Replays" in sev.anchors["EQ"].rationale and "Judgment" in sev.anchors["EQ"].rationale
    only_eq = compose("ai_bubble", complexity="minimal", depth="severe")
    assert set(only_eq.anchors) == {"EQ"} and only_eq.anchors["EQ"].params["loss_scale"] > 1


# --- requires warnings ---------------------------------------------------------


def test_requires_warning_explicit_choice():
    with pytest.warns(ScenarioWarning, match="usd=debasement.*complexity"):
        s = compose("us_bankruptcy", complexity="minimal", usd="debasement")
    assert "USD" not in s.anchors
    assert "usd=debasement" in s.metadata["skipped"]
    with pytest.warns(ScenarioWarning, match="rates_credit=shock"):
        compose("us_bankruptcy", complexity="standard", rates_credit="shock")
    with pytest.warns(ScenarioWarning, match="tech_spread"):
        compose("ai_bubble", complexity="minimal", tech_spread="amplified_1_5x")


def test_defaults_that_do_not_apply_are_silent_but_recorded():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = compose("us_bankruptcy", complexity="minimal")
    assert "rates_credit=shock" in s.metadata["skipped"]


def test_fallback_when_analog_aftermath_unavailable():
    with pytest.warns(ScenarioWarning, match="aftermath=analog"):
        s = compose("ai_bubble", crash_source="parametric", aftermath="analog")
    assert s.aftermath.mode == "stagnation"
    assert s.aftermath.calibrated is False and s.aftermath.rationale
    assert s.metadata["axis_choices"]["aftermath"] == "analog"
    assert "aftermath=analog" in s.metadata["skipped"]


# --- user families: ordering and conflict rule ------------------------------------


def toy_family(**over):
    fam = {
        "name": "toy",
        "base": {
            "calibration_windows": [["2008-09-01", "2009-03-09"]],
            "anchors": {
                "EQ": {"cum_dist": "uniform", "params": {"low": 0.1, "high": 0.5},
                       "calibrated": True, "rationale": "Base."},
            },
            "description": "Toy.",
        },
        "axes": {
            "first": {
                "default": "a",
                "options": {
                    "a": {"summary": "a", "patch": {
                        "crash_length": "1M",
                        "anchors": {"EQ": {"params": {"low": 0.3}, "rationale": "First."}},
                    }},
                    "b": {"summary": "b", "patch": {"horizon": "1Y"}},
                },
            },
            "second": {
                "default": "x",
                "options": {
                    "x": {"summary": "x", "patch": {
                        "crash_length": "6M",
                        "anchors": {"EQ": {"params": {"low": 0.15},
                                           "calibrated": False, "rationale": "Second."}},
                    }},
                    "y": {"summary": "y", "patch": {}},
                },
            },
        },
    }
    fam.update(over)
    return fam


def test_conflict_rule_later_axis_wins_and_rationale_accumulates():
    fam = load_family(toy_family())
    s = compose(fam)
    assert s.crash_length == "6M"  # later axis wins
    assert s.anchors["EQ"].params["low"] == 0.15
    assert s.anchors["EQ"].params["high"] == 0.5  # untouched sibling leaf survives
    assert s.anchors["EQ"].calibrated is False  # any false wins
    assert s.anchors["EQ"].rationale == "Base. First. Second."
    s2 = compose(fam, second="y")
    assert s2.crash_length == "1M" and s2.anchors["EQ"].params["low"] == 0.3
    assert s2.anchors["EQ"].calibrated is True


def test_axis_order_is_declaration_order():
    swapped = toy_family()
    swapped["axes"] = dict(reversed(list(swapped["axes"].items())))
    s = compose(load_family(swapped))
    assert s.crash_length == "1M"  # now `first` is applied last
    assert describe_axes(load_family(swapped))["axis_order"] == ["second", "first"]


def test_family_validation_errors():
    bad = toy_family()
    bad["axes"]["first"]["default"] = "zzz"
    with pytest.raises(SpecError, match="default"):
        load_family(bad)
    bad = toy_family()
    bad["axes"]["first"]["options"]["a"]["requires"] = {"ghost": ["q"]}
    with pytest.raises(SpecError, match="ghost"):
        load_family(bad)
    bad = toy_family()
    bad["axes"]["first"]["options"]["a"]["patch"] = ["not", "a", "dict"]
    with pytest.raises(SpecError, match="patch"):
        load_family(bad)
    bad = toy_family()
    bad["base"]["bogus"] = 1
    with pytest.raises(SpecError, match="bogus"):
        load_family(bad)
    bad = toy_family()
    bad["axes"]["second"]["options"]["y"]["patch"] = {"crash_length": "3Q"}
    with pytest.raises(SpecError, match="duration"):
        load_family(bad)
    with pytest.raises(SpecError, match="name"):
        load_family({"base": {}, "axes": {}})


def test_load_family_from_yaml_path(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text(yaml.safe_dump(toy_family()), encoding="utf-8")
    fam = load_family(p)
    assert fam.name == "toy"
    assert compose(fam).metadata["family"] == "toy"
    assert load_family(str(p)).name == "toy"
    assert grid_size(fam, first=["a", "b"], second=["x", "y"]) == 4


# --- overrides -----------------------------------------------------------------------


def test_overrides_applied_last_and_recorded():
    s = compose("us_bankruptcy", depth="severe", crash_speed="slow",
                overrides={"anchors.EQ.params.median": 0.5, "crash_length": "2M"})
    assert s.anchors["EQ"].params["median"] == 0.5
    assert s.crash_length == "2M"
    assert s.metadata["overrides"] == {"anchors.EQ.params.median": 0.5, "crash_length": "2M"}
    plain = compose("us_bankruptcy", depth="severe", crash_speed="slow")
    assert plain.metadata["overrides"] == {}
    assert s.key != plain.key and s.content_hash != plain.content_hash


def test_override_list_index_and_errors():
    s = compose("us_bankruptcy", overrides={"vol_multiplier.1": 5.0})
    assert s.vol_multiplier == (3.0, 5.0)
    with pytest.raises(SpecError, match="anchors.EQ.params.medain"):
        compose("us_bankruptcy", overrides={"anchors.EQ.params.medain": 0.5})
    with pytest.raises(SpecError, match="override path"):
        compose("us_bankruptcy", overrides={"anchors.NOPE.params.median": 0.5})
    with pytest.raises(SpecError, match="override path"):
        compose("us_bankruptcy", overrides={"metadata": {}})
    with pytest.raises(SpecError, match="duration"):
        compose("us_bankruptcy", overrides={"crash_length": "3Q"})


# --- provenance --------------------------------------------------------------------------


def test_provenance_key_hash():
    s = compose("us_bankruptcy", usd="haven", aftermath="v_recovery")
    md = s.metadata
    assert md["family"] == "us_bankruptcy"
    assert md["axis_choices"] == {
        "complexity": "full", "usd": "haven", "aftermath": "v_recovery",
        "rates_credit": "shock", "depth": "base", "crash_speed": "base",
    }
    assert md["overrides"] == {}
    again = compose("us_bankruptcy", usd="haven", aftermath="v_recovery")
    assert again == s and again.key == s.key and again.content_hash == s.content_hash
    assert s.name == "us_bankruptcy" and s.key.startswith("us_bankruptcy_")
    other = compose("us_bankruptcy", usd="debasement", aftermath="v_recovery")
    assert other.key != s.key and other.content_hash != s.content_hash
    assert len(s.content_hash) == 64


def test_metadata_survives_yaml_with_tuple_override():
    s = compose("us_bankruptcy", overrides={"vol_multiplier": (2.0, 3.0)})
    assert load_spec(yaml.safe_load(dump_spec(s))) == s


# --- aliases ----------------------------------------------------------------------------------

ALIASES = {
    "ai_bubble_worst": ("ai_bubble", dict(complexity="standard", crash_source="analog_exact",
                                          aftermath="analog", tech_spread="as_2000",
                                          depth="as_2000")),
    "ai_bubble_recovery": ("ai_bubble", dict(complexity="standard", crash_source="analog_exact",
                                             aftermath="v_recovery", tech_spread="as_2000",
                                             depth="as_2000")),
}
for _usd, _aft in itertools.product(("debasement", "haven"), ("prolonged_bear", "v_recovery")):
    for _suffix, _rc in (("", "shock"), ("_history_consistent", "history_consistent")):
        ALIASES[f"us_bankruptcy_{_usd}_{_aft}{_suffix}"] = (
            "us_bankruptcy",
            dict(complexity="full", usd=_usd, aftermath=_aft, rates_credit=_rc,
                 depth="base", crash_speed="base"),
        )


def test_list_scenarios_returns_aliases():
    assert sorted(cp.list_scenarios()) == sorted(ALIASES)
    assert len(ALIASES) == 10


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_alias_equals_compose(alias):
    family, axes = ALIASES[alias]
    composed = compose(family, **axes)
    loaded = cp.load_scenario(alias)
    assert loaded.key == alias
    assert dataclasses.replace(loaded, name=composed.name, variant=composed.variant) == composed
    assert loaded.content_hash == composed.content_hash
    assert loaded.metadata == composed.metadata


def test_old_standalone_files_removed():
    from pathlib import Path

    root = Path(cp.__file__).parent
    assert not (root / "scenarios").exists()
    assert sorted(p.stem for p in (root / "families").glob("*.yaml")) == cp.list_families()


def test_unknown_alias_keyerror():
    with pytest.raises(KeyError):
        cp.load_scenario("us_bankruptcy_debasement")  # old unsplit name is gone


# --- property test -------------------------------------------------------------------------------


def _choice_strategy(family):
    axes = describe_axes(family)["axes"]
    return st.fixed_dictionaries(
        {a: st.sampled_from(sorted(v["options"])) for a, v in axes.items()}
    )


@settings(max_examples=60, deadline=None)
@given(choices=_choice_strategy("us_bankruptcy"))
def test_property_us_any_choice_composes(choices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        s = compose("us_bankruptcy", **choices)
    assert s.metadata["axis_choices"] == choices
    assert load_spec(yaml.safe_load(dump_spec(s))) == s


@settings(max_examples=60, deadline=None)
@given(choices=_choice_strategy("ai_bubble"))
def test_property_ai_any_choice_composes(choices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScenarioWarning)
        s = compose("ai_bubble", **choices)
    assert s.metadata["axis_choices"] == choices
    assert load_spec(yaml.safe_load(dump_spec(s))) == s

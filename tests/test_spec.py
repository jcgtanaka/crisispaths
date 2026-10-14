import copy

import pytest
import yaml

import crisispaths as cp
from crisispaths.spec import (
    AftermathSpec,
    FactorShock,
    ScenarioSpec,
    SpecError,
    dump_spec,
    load_spec,
)

ALL = [
    "ai_bubble_worst",
    "ai_bubble_recovery",
    "us_bankruptcy_debasement_prolonged_bear",
    "us_bankruptcy_debasement_v_recovery",
    "us_bankruptcy_haven_prolonged_bear",
    "us_bankruptcy_haven_v_recovery",
    "us_bankruptcy_debasement_prolonged_bear_history_consistent",
    "us_bankruptcy_debasement_v_recovery_history_consistent",
    "us_bankruptcy_haven_prolonged_bear_history_consistent",
    "us_bankruptcy_haven_v_recovery_history_consistent",
]


def minimal():
    return {
        "name": "toy",
        "variant": "worst",
        "label": "main_case",
        "calibration_windows": [["2008-09-01", "2009-03-09"]],
        "anchors": {"EQ": {"cum_dist": "fixed", "params": {"value": -0.3}}},
    }


def test_list_scenarios_has_all():
    assert sorted(cp.list_scenarios()) == sorted(ALL)


@pytest.mark.parametrize("name", ALL)
def test_packaged_roundtrip(name):
    spec = cp.load_scenario(name)
    assert spec.key == name
    again = load_spec(yaml.safe_load(dump_spec(spec)))
    assert again == spec


def test_dump_to_path_and_load(tmp_path):
    spec = cp.load_scenario("ai_bubble_worst")
    p = tmp_path / "s.yaml"
    dump_spec(spec, p)
    assert load_spec(p) == spec
    assert load_spec(str(p)) == spec


def test_ai_bubble_content():
    w = cp.load_scenario("ai_bubble_worst")
    assert "2000-2002 replay, which is itself a 31-month bear: IXIC -78%, SPX -49%" in (
        w.description
    )
    assert w.label == "main_case"
    assert set(w.anchors) == {"EQ", "TECH"}
    assert all(a.cum_dist == "analog" for a in w.anchors.values())
    assert w.analog_window == ("2000-03-10", "2002-10-09")
    assert w.aftermath.mode == "analog"
    r = cp.load_scenario("ai_bubble_recovery")
    assert r.label == "COUNTERFACTUAL"
    assert r.aftermath.mode == "v_recovery"


def test_us_bankruptcy_content():
    d = cp.load_scenario("us_bankruptcy_debasement_prolonged_bear")
    assert [tuple(x) for x in d.calibration_windows] == [
        ("2008-09-01", "2009-03-09"),
        ("2020-02-19", "2020-03-23"),
        ("2011-07-22", "2011-10-03"),
    ]
    eq = d.anchors["EQ"]
    assert eq.cum_dist == "lognormal_loss"
    assert eq.params["median"] == 0.40 and eq.params["sigma"] == 0.276
    assert eq.params["cap"] == 0.95
    assert d.anchors["USD"].params["low"] == -0.20
    assert d.anchors["USD"].params["high"] == -0.10
    h = cp.load_scenario("us_bankruptcy_haven_v_recovery")
    assert h.anchors["USD"].params["low"] == 0.05
    assert h.anchors["USD"].params["high"] == 0.10
    assert d.crash_length == "3M" and d.horizon == "2Y"
    assert d.vol_multiplier == (3.0, 4.0)
    assert d.factor_df == 3.0 and d.copula_df == (10.0, 4.0)
    assert d.aftermath.mode == "prolonged_bear" and d.aftermath.bear_annual_drift == -0.20
    assert h.aftermath.mode == "v_recovery"
    assert h.aftermath.recovery_fraction == (0.5, 0.8)
    assert h.aftermath.recovery_period == "9M"


def test_us_bankruptcy_flags():
    d = cp.load_scenario("us_bankruptcy_debasement_prolonged_bear")
    assert d.anchors["EQ"].calibrated is False and d.anchors["EQ"].rationale
    assert d.anchors["USD"].calibrated is False
    assert d.anchors["HAVEN"].calibrated is False
    for f in ("RATES", "CREDIT"):
        a = d.anchors[f]
        assert a.calibrated is False
        assert "judgment" in a.rationale.lower() and "rallied" in a.rationale
    assert d.aftermath.calibrated is False
    for key in ("factor_df", "copula_df"):
        assert d.param_meta[key]["calibrated"] is False
        assert d.param_meta[key]["rationale"]


def test_history_consistent_unanchors_rates_credit():
    for n in ("us_bankruptcy_debasement_prolonged_bear_history_consistent",
              "us_bankruptcy_haven_v_recovery_history_consistent"):
        s = cp.load_scenario(n)
        assert "RATES" not in s.anchors and "CREDIT" not in s.anchors
        assert {"EQ", "USD", "HAVEN"} <= set(s.anchors)


def test_haven_phases():
    d = cp.load_scenario("us_bankruptcy_debasement_prolonged_bear")
    p = d.anchors["HAVEN"].params
    assert p["liquidation_length"] == "4W"
    assert (p["low"], p["high"]) == (-0.25, -0.10)
    assert (p["post_drift_low"], p["post_drift_high"]) == (0.15, 0.30)
    h = cp.load_scenario("us_bankruptcy_haven_v_recovery")
    ph = h.anchors["HAVEN"].params
    assert (ph["post_drift_low"], ph["post_drift_high"]) == (-0.05, 0.0)


def test_unknown_scenario():
    with pytest.raises(KeyError):
        cp.load_scenario("nope")


def test_defaults():
    s = load_spec(minimal())
    assert s.crash_length == "3M" and s.horizon == "2Y" and s.onset == "0D"
    assert s.vol_multiplier == (3.0, 4.0) and s.copula_df == (10.0, 4.0)
    assert s.analog_mode == "exact"
    assert s.aftermath == AftermathSpec()
    assert s.anchors["EQ"] == FactorShock("fixed", {"value": -0.3})


def test_frozen():
    s = load_spec(minimal())
    with pytest.raises(AttributeError):
        s.name = "x"  # type: ignore[misc]


def mutate(**kw):
    d = copy.deepcopy(minimal())
    d.update(kw)
    return d


@pytest.mark.parametrize(
    "data, msg",
    [
        (mutate(label="maybe"), "label"),
        (mutate(crash_length="3Q"), "duration"),
        (mutate(horizon="soon"), "duration"),
        (mutate(onset=["0D", "xx"]), "duration"),
        (mutate(analog_mode="magic"), "analog_mode"),
        (mutate(aftermath={"mode": "boom"}), "aftermath mode"),
        (mutate(aftermath={"recovery_period": "9X"}), "duration"),
        (mutate(anchors={"EQ": {"cum_dist": "gauss", "params": {}}}), "cum_dist"),
        (mutate(anchors={"EQ": {"cum_dist": "uniform", "params": {"low": 1}}}), "high"),
        (mutate(anchors={"EQ": {"cum_dist": "uniform", "params": {"low": 2, "high": 1}}}), "low"),
        (mutate(calibration_windows=[["2020-01-01"]]), "window"),
        (mutate(calibration_windows=[["2020-03-01", "2020-01-01"]]), "window"),
        (mutate(vol_multiplier=[4.0, 3.0]), "vol_multiplier"),
        (mutate(factor_df=0.5), "factor_df"),
        (mutate(bogus=1), "unknown"),
        (mutate(name=None), "name"),
    ],
)
def test_validation_errors(data, msg):
    if data.get("name", 1) is None:
        del data["name"]
    with pytest.raises(SpecError, match=msg):
        load_spec(data)


def test_spec_is_scenariospec_instance():
    assert isinstance(cp.load_scenario("ai_bubble_worst"), ScenarioSpec)

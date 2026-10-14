"""REPORT.md anatomy: ordering, honesty blocks, sweeps, no averaging across scenarios."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from crisispaths import report
from crisispaths.report import BANNER, write_report
from crisispaths.runner import run_mc, run_sweep

STRAT = "helpers_strategies:sma_trend"
SWEEP_AXES = {"aftermath": ["prolonged_bear", "v_recovery"], "depth": ["as_2000", "severe"],
              "crash_source": "parametric", "complexity": "minimal"}


@pytest.fixture(scope="module")
def prices(market):
    return market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]


@pytest.fixture(scope="module")
def sweep_dir(prices, market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep")
    tab = run_sweep(prices, STRAT, "ai_bubble", SWEEP_AXES, n_paths=40, seed=1, clock=clock,
                    factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0,
                    cost_multiplier=3.0)
    return out, tab


@pytest.fixture(scope="module")
def single_dir(prices, market, clock, tmp_path_factory):
    out = tmp_path_factory.mktemp("single")
    tab = run_mc(prices, STRAT, "ai_bubble_worst", n_paths=30, seed=2, clock=clock,
                 factors=market["factors"], out_dir=out, workers=1, costs_bps=5.0)
    return out, tab


def _text(path):
    return path.read_text(encoding="utf-8")


def test_banner_is_first_and_order_is_fixed(sweep_dir, tmp_path):
    out, tab = sweep_dir
    path = write_report(tab, tmp_path, n_boot=100)
    assert path.name == "REPORT.md"
    t = _text(path)
    heads = ["## Caveats", "## Scenario provenance", "## Factor loadings", "## Results",
             "## Variant comparison"]
    pos = [t.index(BANNER)] + [t.index(h) for h in heads]
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    assert t.index(BANNER) < 400  # right under the title, before any number
    assert BANNER == ("Conditional on this scenario occurring. These are what-ifs, not "
                      "forecasts. No probability is attached to any scenario.")


def test_caveats_content(sweep_dir, tmp_path):
    t = _text(write_report(sweep_dir[1], tmp_path, n_boot=100))
    caveats = t[t.index("## Caveats"): t.index("## Scenario provenance")]
    assert "40 paths" in caveats
    assert "cost_multiplier" in caveats and "3.0" in caveats
    assert "censored" in caveats or "did not recover" in caveats or "all paths recovered" in caveats
    assert "unstable" in caveats  # < 200 paths


def test_analog_exact_conditionality_is_stated(single_dir, tmp_path):
    t = _text(write_report(single_dir[1], tmp_path, n_boot=100))
    caveats = t[t.index("## Caveats"): t.index("## Scenario provenance")]
    assert "single history" in caveats


def test_judgment_parameters_listed(sweep_dir, tmp_path):
    out, tab = sweep_dir
    t = _text(write_report(out, tmp_path, n_boot=100))
    prov = t[t.index("## Scenario provenance"): t.index("## Factor loadings")]
    assert "JUDGMENT" in prov
    man = json.loads(next(out.glob("manifest_*.json")).read_text())
    flagged = [k for k, v in man["spec"]["anchors"].items() if v.get("calibrated") is False]
    flagged += ["aftermath"] if man["spec"]["aftermath"].get("calibrated") is False else []
    assert flagged
    for name in flagged:
        assert name in prov
    assert man["spec_content_hash"][:12] in prov


def test_loadings_table(sweep_dir, tmp_path):
    t = _text(write_report(sweep_dir[1], tmp_path, n_boot=100))
    ld = t[t.index("## Factor loadings"): t.index("## Results")]
    for col in ("n_obs", "shrinkage", "unaffected"):
        assert col in ld
    assert "A045" in ld and "NOISE" in ld


def test_results_have_benchmark_cis(single_dir, tmp_path):
    t = _text(write_report(single_dir[1], tmp_path, n_boot=100))
    res = t[t.index("## Results"):]
    assert "benchmark" in res.lower()
    assert re.search(r"\[-?\d+\.\d+, -?\d+\.\d+\]", res)  # a CI
    assert "Kaplan-Meier" in res
    assert "## Variant comparison" not in t  # a single scenario has no sweep section


def test_variant_comparison_shows_axis_ranges(sweep_dir, tmp_path):
    t = _text(write_report(sweep_dir[1], tmp_path, n_boot=100))
    var = t[t.index("## Variant comparison"):]
    assert "v_recovery" in var and "prolonged_bear" in var and "severe" in var
    assert "judgment" in var.lower()


def test_no_expected_return_across_scenarios(sweep_dir, tmp_path):
    out, tab = sweep_dir
    t = _text(write_report(tab, tmp_path, n_boot=100))
    low = t.lower()
    for bad in ("expected return", "average across", "mean across", "probability-weighted"):
        assert bad not in low
    pooled_mean = tab["terminal_wealth"].mean()
    pooled_median = tab["terminal_wealth"].median()
    for v in (pooled_mean, pooled_median):
        # a pooled cross-scenario figure would be printed with 4 decimals somewhere
        assert f"{v:.4f}" not in t
    per_scen = tab.groupby("scenario")["terminal_wealth"].median()
    assert f"{per_scen.mean():.4f}" not in t


def test_report_from_dir_equals_from_frame(single_dir, tmp_path):
    out, tab = single_dir
    a = write_report(tab, tmp_path / "a", n_boot=100)
    b = write_report(out, tmp_path / "b", n_boot=100)
    assert _text(a) == _text(b)


def test_plots_written_or_graceful_fallback(single_dir, tmp_path, monkeypatch):
    out, tab = single_dir
    monkeypatch.setattr(report, "_pyplot", lambda: None)
    p = write_report(tab, tmp_path / "noplot", n_boot=100)
    assert not list((tmp_path / "noplot").glob("*.png"))
    assert "matplotlib" in _text(p)
    monkeypatch.undo()
    pytest.importorskip("matplotlib")
    d = tmp_path / "plots"
    p = write_report(tab, d, n_boot=100)
    names = {f.name.split("_")[0] for f in d.glob("*.png")}
    assert {"fan", "drawdown", "worst10", "recovery"} <= names
    assert "![" in _text(p)


def test_intraday_warning_rule():
    assert report.intraday_warning(252.0, 500.0) is None
    assert report.intraday_warning(2520.0, 1.0) is None
    assert "optimistic" in report.intraday_warning(2520.0, 500.0)


def test_partial_run_is_flagged(prices, market, clock, tmp_path):
    out = tmp_path / "o"
    run_mc(prices, STRAT, "ai_bubble_worst", n_paths=5, seed=0, clock=clock,
           factors=market["factors"], out_dir=out, workers=1, costs_bps=1.0)
    mf = next(out.glob("manifest_*.json"))
    m = json.loads(mf.read_text())
    m["n_paths"] = 9
    mf.write_text(json.dumps(m))
    t = _text(write_report(out, tmp_path / "r", n_boot=50))
    assert "5 of 9" in t


def test_empty_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        write_report(tmp_path, tmp_path / "r")
    with pytest.raises(ValueError):
        write_report(pd.DataFrame({"a": [1]}), tmp_path / "r")


# --- path sanity and imputed loadings ------------------------------------------------------


def _synthetic(crash_a=-0.2, crash_b=0.15, imputed=("B",)):
    """Two assets on one anchored factor EQ that falls 30% in the crash leg."""
    n = 50
    tab = pd.DataFrame({
        "crashret_A": np.full(n, crash_a), "horret_A": np.full(n, 0.05),
        "crashret_B": np.full(n, crash_b), "horret_B": np.full(n, 0.4),
        "fcrash_EQ": np.full(n, -0.3), "fcrash_USD": np.full(n, 0.5),
    })
    man = {
        "key": "k", "assets": ["A", "B"], "factors": ["EQ", "USD"],
        "spec": {"anchors": {"EQ": {"cum_dist": "fixed"}}},
        "loadings": {
            "regimes": ["calm", "stress"], "assets": ["A", "B"], "factors": ["EQ", "USD"],
            "betas": {"calm": [[1.0, 0.0], [0.8, 0.0]], "stress": [[1.0, 0.0], [1.6, 0.0]]},
            "std_errors": {}, "n_obs": {}, "shrinkage": {}, "unaffected": [],
            "imputed": list(imputed), "warnings": [],
            "imputation": {"B": {"amplification": 2.0, "neighbours": ["A"], "clipped": False,
                                 "degenerate": False}},
        },
    }
    return man, tab


def test_path_sanity_flags_sign_mismatch_and_imputed():
    man, tab = _synthetic()
    rows = {r["asset"]: r for r in report.path_sanity(man, tab)}
    assert rows["A"]["crash_median"] == pytest.approx(-0.2)
    assert rows["A"]["horizon_median"] == pytest.approx(0.05)
    assert rows["A"]["expected"] == pytest.approx(-0.3)  # only the anchored factor counts
    assert not rows["A"]["warning"] and not rows["A"]["imputed"]
    assert rows["B"]["warning"] and "opposite sign" in rows["B"]["warning"]
    assert rows["B"]["imputed"] and rows["B"]["amplification"] == 2.0


def test_path_sanity_quiet_when_consistent():
    man, tab = _synthetic(crash_b=-0.4, imputed=())
    rows = report.path_sanity(man, tab)
    assert not any(r["warning"] for r in rows) and not any(r["imputed"] for r in rows)


def test_path_sanity_section_and_imputed_row_in_report(single_dir, tmp_path):
    out, _ = single_dir
    src = tmp_path / "src"
    src.mkdir()
    for f in out.iterdir():
        (src / f.name).write_bytes(f.read_bytes())
    mf = next(src.glob("manifest_*.json"))
    m = json.loads(mf.read_text())
    asset = m["loadings"]["assets"][0]
    m["loadings"]["imputed"] = [asset]
    m["loadings"].setdefault("imputation", {})[asset] = {
        "amplification": 1.37, "neighbours": ["X"], "clipped": False, "degenerate": False}
    mf.write_text(json.dumps(m))
    t = _text(write_report(src, tmp_path / "r", n_boot=50, plots=False))
    sec = t[t.index("## Path sanity"): t.index("## Results")]
    assert t.index("## Factor loadings") < t.index("## Path sanity") < t.index("## Results")
    assert asset in sec and "IMPUTED" in sec
    ld = t[t.index("## Factor loadings"): t.index("## Path sanity")]
    want = "IMPUTED (no data in stress windows; borrowed from neighbours, amplification s=1.37)"
    assert want in ld

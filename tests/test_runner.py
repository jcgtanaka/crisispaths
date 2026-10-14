"""Resumable parallel runner: resume, regeneration, pins, partial lines, spawn, sweeps."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import crisispaths as cp
from crisispaths import runner
from crisispaths.backtest import run_one_path
from crisispaths.runner import PinMismatchError, run_mc, run_sweep
from helpers_strategies import sma_trend

STRAT = "helpers_strategies:sma_trend"


@pytest.fixture(scope="module")
def prices(market):
    return market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]]


@pytest.fixture(scope="module")
def factors(market):
    return market["factors"]


@pytest.fixture(scope="module")
def cal(prices, factors, clock, spec_ai):
    return cp.calibrate(prices, spec_ai, clock=clock, factors=factors)


def _csv(out, spec):
    return out / f"results_{spec.key}.csv"


def _frame(path):
    return pd.read_csv(path).drop(columns=lambda c: c.startswith("eq_"), errors="ignore")


def test_runner_matches_backtest_rows(prices, clock, spec_ai, cal, tmp_path):
    ref = cp.backtest(sma_trend, prices, spec_ai, n_paths=3, seed=5, clock=clock,
                      calibration=cal, costs_bps=4.0, n_boot=20)
    tab = run_mc(prices, sma_trend, spec_ai, n_paths=3, seed=5, clock=clock, calibration=cal,
                 costs_bps=4.0, out_dir=tmp_path, workers=1)
    got = tab.drop(columns=["scenario"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, ref.table, check_exact=False, rtol=1e-12)
    assert set(tab["scenario"]) == {spec_ai.key}


def test_output_files_and_pin(prices, clock, spec_ai, cal, tmp_path):
    run_mc(prices, STRAT, spec_ai, n_paths=2, seed=3, clock=clock, calibration=cal,
           out_dir=tmp_path, workers=1)
    pin = json.loads((tmp_path / f"pin_{spec_ai.key}.json").read_text())
    assert pin["spec_content_hash"] == spec_ai.content_hash
    assert pin["calibration_hash"] == cp.calibration_hash(cal)
    assert pin["seed"] == 3
    assert pin["package_version"] == cp.__version__
    raw = _csv(tmp_path, spec_ai).read_text().splitlines()
    assert len(raw) == 3 and raw[0].startswith("path,")


def test_resume_equals_uninterrupted(prices, clock, spec_ai, cal, tmp_path):
    full, half = tmp_path / "full", tmp_path / "half"
    kw = dict(seed=11, clock=clock, calibration=cal, costs_bps=2.0, workers=1)
    run_mc(prices, STRAT, spec_ai, n_paths=6, out_dir=full, **kw)
    run_mc(prices, STRAT, spec_ai, n_paths=3, out_dir=half, **kw)  # the "kill"
    calls = []
    tab = run_mc(prices, STRAT, spec_ai, n_paths=6, out_dir=half,
                 progress=lambda key, done, total: calls.append(done), **kw)
    assert _csv(half, spec_ai).read_bytes() == _csv(full, spec_ai).read_bytes()
    assert len(tab) == 6 and sorted(tab["path"]) == list(range(6))
    assert calls == [4, 5, 6]  # only the missing paths ran


def test_path_alone_equals_path_in_batch(prices, clock, spec_ai, cal, tmp_path):
    kw = dict(seed=2, clock=clock, calibration=cal, workers=1)
    batch = run_mc(prices, STRAT, spec_ai, n_paths=4, out_dir=tmp_path / "b", **kw)
    one = run_mc(prices, STRAT, spec_ai, n_paths=4, out_dir=tmp_path / "o", **kw)
    row = batch[batch["path"] == 2].iloc[0]
    hist = prices[list(cal.assets)]
    alone, _ = run_one_path(
        sma_trend, hist, cal, spec_ai, 2, 2, clock, costs_bps=0.0, cost_multiplier=1.0
    )
    assert alone["terminal_wealth"] == row["terminal_wealth"]
    pd.testing.assert_frame_equal(batch, one)


def test_pin_mismatch_raises(prices, clock, spec_ai, cal, tmp_path):
    kw = dict(clock=clock, calibration=cal, workers=1, out_dir=tmp_path)
    run_mc(prices, STRAT, spec_ai, n_paths=2, seed=1, **kw)
    with pytest.raises(PinMismatchError, match="seed"):
        run_mc(prices, STRAT, spec_ai, n_paths=3, seed=2, **kw)
    with pytest.raises(PinMismatchError, match="costs_bps"):
        run_mc(prices, STRAT, spec_ai, n_paths=3, seed=1, costs_bps=9.0, **kw)
    pin_path = tmp_path / f"pin_{spec_ai.key}.json"
    pin = json.loads(pin_path.read_text())
    pin["package_version"] = "9.9.9"
    pin_path.write_text(json.dumps(pin))
    with pytest.raises(PinMismatchError, match="package_version"):
        run_mc(prices, STRAT, spec_ai, n_paths=3, seed=1, **kw)
    # --no-resume starts over instead of raising
    tab = run_mc(prices, STRAT, spec_ai, n_paths=2, seed=7, resume=False, **kw)
    assert len(tab) == 2
    assert json.loads(pin_path.read_text())["seed"] == 7


def test_partial_last_line_is_discarded(prices, clock, spec_ai, cal, tmp_path):
    kw = dict(seed=4, clock=clock, calibration=cal, workers=1)
    run_mc(prices, STRAT, spec_ai, n_paths=4, out_dir=tmp_path / "ref", **kw)
    run_mc(prices, STRAT, spec_ai, n_paths=3, out_dir=tmp_path / "cut", **kw)
    f = _csv(tmp_path / "cut", spec_ai)
    with open(f, "ab") as fh:
        fh.write(b"3,0.93,0.0")  # killed mid-write: no newline, too few fields
    run_mc(prices, STRAT, spec_ai, n_paths=4, out_dir=tmp_path / "cut", **kw)
    assert f.read_bytes() == _csv(tmp_path / "ref", spec_ai).read_bytes()


def test_partial_header_is_discarded(prices, clock, spec_ai, cal, tmp_path):
    f = _csv(tmp_path, spec_ai)
    f.write_text("path,terminal_we")
    tab = run_mc(prices, STRAT, spec_ai, n_paths=2, seed=0, clock=clock, calibration=cal,
                 out_dir=tmp_path, workers=1)
    assert len(tab) == 2


def test_spawn_workers_equal_in_process(prices, clock, spec_ai, cal, tmp_path):
    kw = dict(n_paths=4, seed=8, clock=clock, calibration=cal, costs_bps=3.0)
    a = run_mc(prices, STRAT, spec_ai, out_dir=tmp_path / "w1", workers=1, **kw)
    b = run_mc(prices, STRAT, spec_ai, out_dir=tmp_path / "w2", workers=2, **kw)
    pd.testing.assert_frame_equal(a, b)
    f1, f2 = _csv(tmp_path / "w1", spec_ai), _csv(tmp_path / "w2", spec_ai)
    assert f1.read_bytes() == f2.read_bytes()


def test_callable_strategy_resolution(prices, clock, spec_ai, cal, tmp_path):
    # a top-level function of an importable module is accepted for workers > 1
    run_mc(prices, sma_trend, spec_ai, n_paths=2, clock=clock, calibration=cal,
           out_dir=tmp_path / "a", workers=2)
    with pytest.raises(ValueError, match="module:fn"):
        run_mc(prices, lambda p: p * 0 + 0.2, spec_ai, n_paths=2, clock=clock, calibration=cal,
               out_dir=tmp_path / "b", workers=2)
    with pytest.raises(ValueError, match="module:fn"):
        run_mc(prices, "no_colon_here", spec_ai, n_paths=2, clock=clock, calibration=cal,
               out_dir=tmp_path / "c", workers=1)
    with pytest.raises(ValueError, match="cannot import"):
        run_mc(prices, "helpers_strategies:nope", spec_ai, n_paths=2, clock=clock,
               calibration=cal, out_dir=tmp_path / "d", workers=1)


def test_keyboard_interrupt_keeps_partial_results(prices, clock, spec_ai, cal, tmp_path):
    def stop_after_two(key, done, total):
        if done == 2:
            raise KeyboardInterrupt

    kw = dict(seed=6, clock=clock, calibration=cal, workers=1, out_dir=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run_mc(prices, STRAT, spec_ai, n_paths=5, progress=stop_after_two, **kw)
    assert len(_frame(_csv(tmp_path, spec_ai))) == 2
    tab = run_mc(prices, STRAT, spec_ai, n_paths=5, **kw)
    assert sorted(tab["path"]) == list(range(5))


def test_default_workers():
    assert runner.default_workers(1000, cpu_count=8) == 7
    assert runner.default_workers(3, cpu_count=8) == 3
    assert runner.default_workers(10, cpu_count=1) == 1
    assert runner.default_workers(10, cpu_count=2) == 1


def test_costs_and_multiplier_are_applied(prices, clock, spec_ai, cal, tmp_path):
    kw = dict(n_paths=2, seed=1, clock=clock, calibration=cal, workers=1)
    a = run_mc(prices, STRAT, spec_ai, out_dir=tmp_path / "a", costs_bps=5.0, **kw)
    b = run_mc(prices, STRAT, spec_ai, out_dir=tmp_path / "b", costs_bps=5.0,
               cost_multiplier=3.0, **kw)
    assert (b["total_cost"] > a["total_cost"]).all()
    np.testing.assert_allclose(b["total_cost"], 3 * a["total_cost"], rtol=1e-9)


# --- sweeps --------------------------------------------------------------------------------


def test_sweep_long_table_and_guard(prices, factors, clock, tmp_path):
    axes = {"aftermath": ["prolonged_bear", "stagnation"], "depth": ["as_2000", "severe"],
            "crash_source": "parametric", "complexity": "minimal"}
    tab = run_sweep(prices, STRAT, "ai_bubble", axes, n_paths=2, seed=0, clock=clock,
                    factors=factors, out_dir=tmp_path, workers=1, benchmark=None)
    assert {"aftermath", "depth", "scenario", "path", "merged_choices"} <= set(tab.columns)
    assert tab["scenario"].nunique() == 4 and len(tab) == 8
    assert set(tab["depth"]) == {"as_2000", "severe"}
    with pytest.raises(ValueError, match="max_scenarios") as ei:
        run_sweep(prices, STRAT, "ai_bubble", axes, n_paths=2, seed=0, clock=clock,
                  factors=factors, out_dir=tmp_path / "x", workers=1, max_scenarios=3)
    assert "4" in str(ei.value)
    assert not (tmp_path / "x").exists() or not any((tmp_path / "x").iterdir())


def test_sweep_unknown_axis_is_an_error(prices, factors, clock, tmp_path):
    with pytest.raises(cp.SpecError):
        run_sweep(prices, STRAT, "ai_bubble", {"nope": ["a", "b"]}, n_paths=2, clock=clock,
                  factors=factors, out_dir=tmp_path, workers=1)


def test_run_mc_accepts_names_and_lists(prices, factors, clock, tmp_path):
    tab = run_mc(prices, STRAT, ["ai_bubble_worst", "ai_bubble_recovery"], n_paths=1,
                 clock=clock, factors=factors, out_dir=tmp_path, workers=1)
    assert set(tab["scenario"]) == {"ai_bubble_worst", "ai_bubble_recovery"}
    assert os.path.exists(tmp_path / "results_ai_bubble_worst.csv")


def test_path_diagnostics_columns(prices, clock, spec_ai, cal, tmp_path):
    from crisispaths import generate_path

    tab = run_mc(prices, STRAT, spec_ai, n_paths=2, seed=3, clock=clock, calibration=cal,
                 out_dir=tmp_path, workers=1)
    assert not any(c.startswith(("eq_", "crashret_", "horret_", "fcrash_")) for c in tab.columns)
    raw = runner.load_results(tmp_path, spec_ai.key, keep_eq=True)
    p = generate_path(cal, spec_ai, 1, 3, clock)
    crash = p.phase == 1
    for a in cal.assets:
        want_c = np.expm1(p.returns[a].to_numpy()[crash].sum())
        want_h = np.expm1(p.returns[a].to_numpy().sum())
        assert raw.loc[1, f"crashret_{a}"] == pytest.approx(want_c, rel=1e-9)
        assert raw.loc[1, f"horret_{a}"] == pytest.approx(want_h, rel=1e-9)
    for f in cal.factors:
        assert raw.loc[1, f"fcrash_{f}"] == pytest.approx(
            p.factors[f].to_numpy()[crash].sum(), rel=1e-9, abs=1e-12)


def test_calm_drift_is_pinned(prices, factors, clock, spec_ai, tmp_path):
    kw = dict(n_paths=2, seed=0, clock=clock, factors=factors, out_dir=tmp_path, workers=1)
    run_mc(prices, STRAT, spec_ai, calm_drift="zero", **kw)
    with pytest.raises(PinMismatchError, match="calibration_hash"):
        run_mc(prices, STRAT, spec_ai, calm_drift="historical", **kw)
    man = json.loads((tmp_path / f"manifest_{spec_ai.key}.json").read_text())
    assert man["calm_drift"] == "zero"


def test_calm_drift_conflicts_with_given_calibration(prices, clock, spec_ai, cal, tmp_path):
    with pytest.raises(ValueError, match="calm_drift"):
        run_mc(prices, STRAT, spec_ai, n_paths=1, clock=clock, calibration=cal,
               calm_drift="historical", out_dir=tmp_path, workers=1)

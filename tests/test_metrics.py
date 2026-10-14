"""Path metrics, benchmark-relative metrics, Kaplan-Meier, bootstrap CIs and the summary."""

import numpy as np
import pandas as pd
import pytest

import crisispaths as cp
from crisispaths.metrics import (
    bootstrap_ci,
    cvar,
    kaplan_meier,
    path_metrics,
    relative_metrics,
    summarize,
    wilson_interval,
)


def eq(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)


def test_drawdown_and_recovery_hand_computed():
    m = path_metrics(eq([1, 1.2, 0.9, 0.6, 0.9, 1.3]), history_peak=None, ruin_level=0.65)
    assert m["terminal_wealth"] == pytest.approx(1.3)
    assert m["terminal_log_return"] == pytest.approx(np.log(1.3))
    assert m["max_drawdown_from_peak"] == pytest.approx(0.5)  # 1.2 -> 0.6
    assert m["max_drawdown_from_start"] == pytest.approx(0.5)
    assert m["loss_from_start"] == pytest.approx(0.4)
    assert m["time_to_trough_bars"] == 3
    assert m["recovered"] is True
    assert m["time_to_recover_bars"] == 2  # trough at 3, regains 1.2 at bar 5
    assert m["recovery_time_bars"] == 2
    assert m["ruined"] is True and m["time_to_ruin_bars"] == 3
    assert m["horizon_bars"] == 5


def test_censored_recovery_is_nan_with_censor_time():
    m = path_metrics(eq([1, 1.1, 0.8, 0.9, 0.95]), history_peak=None)
    assert m["max_drawdown_from_peak"] == pytest.approx(1 - 0.8 / 1.1)
    assert m["time_to_trough_bars"] == 2
    assert m["recovered"] is False
    assert np.isnan(m["time_to_recover_bars"])
    assert m["recovery_time_bars"] == 2  # censored at horizon: 4 - 2 bars observed
    assert m["ruined"] is False and np.isnan(m["time_to_ruin_bars"])


def test_no_drawdown_counts_as_recovered_at_zero():
    m = path_metrics(eq([1, 1.1, 1.2]), history_peak=None)
    assert m["max_drawdown_from_peak"] == 0.0
    assert m["recovered"] is True and m["time_to_recover_bars"] == 0
    assert m["time_to_trough_bars"] == 0


def test_history_peak_changes_peak_drawdown_not_start_drawdown():
    m = path_metrics(eq([1, 1.1, 1.2]), history_peak=1.5)
    assert m["max_drawdown_from_start"] == 0.0
    assert m["max_drawdown_from_peak"] == pytest.approx(1 - 1.0 / 1.5)
    assert m["recovered"] is False  # never regains 1.5 inside the horizon
    assert m["recovery_time_bars"] == 2


def test_horizon_truncates_observation():
    e = [1, 0.5, 1.0, 2.0]
    assert path_metrics(eq(e), history_peak=None, horizon_bars=2)["time_to_recover_bars"] == 1
    m = path_metrics(eq(e), history_peak=None, horizon_bars=1)
    assert m["recovered"] is False and m["recovery_time_bars"] == 0
    assert m["terminal_wealth"] == pytest.approx(0.5)


def test_path_metrics_validation():
    with pytest.raises(ValueError, match="start"):
        path_metrics(eq([0.0, 1.0]), history_peak=None)
    with pytest.raises(ValueError, match="NaN|finite"):
        path_metrics(eq([1.0, np.nan]), history_peak=None)


def test_relative_metrics():
    s = path_metrics(eq([1, 1.5, 0.9]), history_peak=None)
    b = path_metrics(eq([1, 1.0, 0.8]), history_peak=None)
    r = relative_metrics(s, b)
    assert r["excess_terminal_wealth"] == pytest.approx(0.1)
    assert r["excess_max_drawdown_from_peak"] == pytest.approx(0.4 - 0.2)


# --- Kaplan-Meier -------------------------------------------------------------------------


def test_kaplan_meier_hand_computed_with_censoring():
    km = kaplan_meier([1, 2, 2, 3, 5], [True, True, False, True, False])
    # t=1: 5 at risk, 1 event -> .8 ; t=2: 4 at risk, 1 event (+1 censored) -> .6 ;
    # t=3: 2 at risk, 1 event -> .3 ; t=5 censored
    got = dict(zip(km["time"], km["survival"], strict=True))
    assert got[1] == pytest.approx(0.8)
    assert got[2] == pytest.approx(0.6)
    assert got[3] == pytest.approx(0.3)
    assert km.attrs["share_recovered"] == pytest.approx(0.7)
    assert km.attrs["median_time"] == 3


def test_kaplan_meier_equals_empirical_share_without_censoring():
    rng = np.random.default_rng(0)
    t = rng.integers(1, 50, 120)
    ev = np.ones(120, bool)
    km = kaplan_meier(t, ev)
    assert km.attrs["share_recovered"] == pytest.approx(1.0)
    assert km["survival"].iloc[-1] == pytest.approx(0.0)
    # S(t) = share with T > t, exactly
    for _, row in km.iterrows():
        assert row["survival"] == pytest.approx((t > row["time"]).mean())
    assert km.attrs["median_time"] == np.sort(t)[int(np.ceil(120 / 2)) - 1]


def test_kaplan_meier_median_undefined_when_survival_stays_high():
    km = kaplan_meier([1, 5, 5, 5, 5], [True, False, False, False, False])
    assert km.attrs["share_recovered"] == pytest.approx(0.2)
    assert np.isnan(km.attrs["median_time"])


# --- bootstrap, Wilson, CVaR -----------------------------------------------------------------


def test_cvar_hand_computed():
    x = np.arange(1, 101, dtype=float)
    assert cvar(x, 0.05, tail="low") == pytest.approx(3.0)
    assert cvar(x, 0.05, tail="high") == pytest.approx(98.0)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3) and hi == pytest.approx(0.1744, abs=1e-3)
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1


def test_bootstrap_deterministic_and_n_boot_respected():
    x = np.random.default_rng(1).normal(size=150)
    a = bootstrap_ci(x, lambda s: s.mean(axis=1), n_boot=300, seed=5, return_replicates=True)
    b = bootstrap_ci(x, lambda s: s.mean(axis=1), n_boot=300, seed=5, return_replicates=True)
    c = bootstrap_ci(x, lambda s: s.mean(axis=1), n_boot=300, seed=6, return_replicates=True)
    assert a[2].shape == (300,)
    np.testing.assert_array_equal(a[2], b[2])
    assert not np.array_equal(a[2], c[2])
    assert a[0] < x.mean() < a[1]


def test_bootstrap_coverage_sanity():
    rng = np.random.default_rng(11)
    hit = 0
    reps = 150
    for r in range(reps):
        x = rng.normal(0.0, 1.0, 100)
        lo, hi = bootstrap_ci(x, lambda s: s.mean(axis=1), n_boot=300, seed=r)
        hit += lo <= 0.0 <= hi
    assert 0.88 <= hit / reps <= 0.99


# --- summarize ----------------------------------------------------------------------------


def table(n=400, seed=0, censor=True):
    rng = np.random.default_rng(seed)
    tw = np.exp(rng.normal(-0.1, 0.3, n))
    dd = np.clip(rng.beta(2, 5, n), 0, 1)
    rec_t = rng.integers(1, 300, n).astype(float)
    recovered = rng.uniform(size=n) < (0.7 if censor else 1.0)
    return pd.DataFrame(
        {
            "terminal_wealth": tw,
            "max_drawdown_from_peak": dd,
            "ruined": tw < 0.6,
            "recovered": recovered,
            "recovery_time_bars": np.where(recovered, rec_t, 200.0),
            "time_to_recover_bars": np.where(recovered, rec_t, np.nan),
            "horizon_bars": 504,
        }
    )


def test_summary_structure_and_values():
    t = table()
    s = summarize(t, alpha=0.05, n_boot=400, seed=1)
    q = s.quantiles
    row = q[(q["metric"] == "terminal_wealth") & (q["quantile"] == 0.5)].iloc[0]
    assert row["value"] == pytest.approx(t["terminal_wealth"].median())
    assert row["ci_low"] <= row["value"] <= row["ci_high"]
    assert set(q["quantile"]) == {0.05, 0.25, 0.5, 0.75, 0.95}
    c = s.cvar["terminal_wealth"]
    k = int(np.ceil(0.05 * len(t)))
    assert c["value"] == pytest.approx(np.sort(t["terminal_wealth"])[:k].mean())
    assert c["ci_low"] <= c["value"] <= c["ci_high"]
    d = s.cvar["max_drawdown_from_peak"]
    assert d["value"] == pytest.approx(np.sort(t["max_drawdown_from_peak"])[-k:].mean())
    r = s.ruin
    assert r["n_ruined"] == int(t["ruined"].sum()) and r["n"] == len(t)
    assert r["ci_low"] <= r["p"] <= r["ci_high"]
    rec = s.recovery
    assert 0 < rec["share_recovered"] <= 1
    assert rec["n_censored"] == int((~t["recovered"]).sum())
    # never a mean of censored times
    naive = t["time_to_recover_bars"].mean()
    assert "mean" not in " ".join(rec)
    assert not np.isnan(rec["median_time"]) or rec["share_recovered"] < 0.5
    assert np.isnan(naive) or rec["median_time"] != naive
    assert s.meta["n_boot"] == 400 and s.meta["seed"] == 1


def test_summary_deterministic_given_seed():
    t = table()
    a = summarize(t, n_boot=200, seed=3)
    b = summarize(t, n_boot=200, seed=3)
    pd.testing.assert_frame_equal(a.quantiles, b.quantiles)
    assert a.cvar == b.cvar and a.ruin == b.ruin


def test_summary_caveats():
    small = summarize(table(n=50), n_boot=100)
    assert any("200" in c for c in small.caveats)
    big = summarize(table(n=400), n_boot=100)
    assert not any("200" in c for c in big.caveats)
    assert not any("single history" in c for c in big.caveats)
    exact = summarize(table(n=400), n_boot=100, analog_exact=True)
    assert any("single history" in c for c in exact.caveats)
    cens = summarize(table(n=400), n_boot=100)
    assert any("censored" in c for c in cens.caveats)


def test_summary_requires_columns():
    with pytest.raises(ValueError, match="missing"):
        summarize(pd.DataFrame({"terminal_wealth": [1.0]}), n_boot=10)


def test_public_exports():
    for name in ("run_strategy", "StrategyResult", "check_lookahead", "LookaheadReport",
                 "path_metrics", "summarize", "Summary", "backtest", "BacktestResult",
                 "buy_and_hold_equal_weight"):
        assert hasattr(cp, name), name

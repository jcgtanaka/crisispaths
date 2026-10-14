"""End-to-end backtest() on the synthetic fixture market."""

import time

import numpy as np
import pandas as pd
import pytest

import crisispaths as cp

TABLE_COLUMNS = {
    "path", "terminal_wealth", "terminal_log_return", "max_drawdown_from_start",
    "max_drawdown_from_peak", "loss_from_start", "time_to_trough_bars", "time_to_recover_bars",
    "recovered", "recovery_time_bars", "ruined", "time_to_ruin_bars", "total_turnover",
    "bench_terminal_wealth", "bench_max_drawdown_from_peak", "excess_terminal_wealth",
    "excess_max_drawdown_from_peak",
}


def sma_trend(prices):
    ma = prices.rolling(50).mean()
    return (prices > ma).astype(float).where(ma.notna()) / prices.shape[1]


def buy_hold(prices):
    return pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)


@pytest.fixture(scope="module")
def prices(market):
    return market["prices"]


def test_ai_bubble_end_to_end(prices, clock, cal_ai):
    res = cp.backtest(sma_trend, prices, "ai_bubble", n_paths=50, seed=4, clock=clock,
                      calibration=cal_ai, n_boot=200)
    assert isinstance(res, cp.BacktestResult)
    assert len(res.table) == 50
    assert TABLE_COLUMNS <= set(res.table.columns)
    assert res.table["terminal_wealth"].notna().all()
    assert res.spec.name == "ai_bubble"
    assert res.meta["seed"] == 4 and res.meta["n_paths"] == 50
    assert res.meta["package_version"] == cp.__version__
    assert res.meta["spec_content_hash"] == res.spec.content_hash
    assert res.meta["spec_metadata"] == res.spec.metadata
    assert len(res.calibration_hash) == 64
    assert any(c for c in res.summary.caveats if "200" in c)  # 50 paths < 200
    assert isinstance(res.summary, cp.Summary)


def test_us_bankruptcy_buy_and_hold_and_determinism(prices, clock, cal_us, spec_us):
    kw = dict(n_paths=30, seed=9, clock=clock, calibration=cal_us, n_boot=100)
    a = cp.backtest(buy_hold, prices, spec_us, **kw)
    b = cp.backtest(buy_hold, prices, "us_bankruptcy", **kw)
    pd.testing.assert_frame_equal(a.table, b.table)
    c = cp.backtest(buy_hold, prices, spec_us, **{**kw, "seed": 10})
    assert not a.table["terminal_wealth"].equals(c.table["terminal_wealth"])
    assert a.calibration_hash == cal_hash(cal_us)
    # equal-weight buy-and-hold vs the equal-weight benchmark: identical paths, zero excess
    np.testing.assert_allclose(a.table["excess_terminal_wealth"].to_numpy(), 0.0, atol=1e-12)
    assert (a.table["terminal_wealth"] < 1.0).mean() > 0.5  # a crash scenario


def cal_hash(cal):
    from crisispaths.backtest import calibration_hash

    return calibration_hash(cal)


def test_analog_exact_caveat_and_leaky_strategy_blocked(prices, clock, cal_ai, spec_ai):
    res = cp.backtest(buy_hold, prices, spec_ai, n_paths=10, seed=1, clock=clock,
                      calibration=cal_ai, n_boot=50)
    exact = spec_ai.analog_mode == "exact"
    assert any("single history" in c for c in res.summary.caveats) == exact

    def leaky(p):
        return (p.shift(-1) > p).astype(float) / p.shape[1]

    with pytest.raises(cp.LookaheadError):
        cp.backtest(leaky, prices, spec_ai, n_paths=2, clock=clock, calibration=cal_ai)


def test_throughput_50_paths_5_assets_504_bars(market, clock, spec_ai):
    p5 = market["prices"].iloc[:, :5]
    fac = market["factors"]
    from conftest import CALM_WINDOWS

    cal5 = cp.calibrate(p5, spec_ai, clock=clock, factors=fac, calm_windows=CALM_WINDOWS)
    t0 = time.perf_counter()
    res = cp.backtest(sma_trend, p5, spec_ai, n_paths=50, seed=0, clock=clock,
                      calibration=cal5, n_boot=500)
    elapsed = time.perf_counter() - t0
    print(f"\nbacktest 50 paths x 5 assets x {res.meta['bars']} bars: {elapsed:.2f}s")
    assert res.meta["bars"] == 504 and len(res.table) == 50
    assert elapsed < 30.0

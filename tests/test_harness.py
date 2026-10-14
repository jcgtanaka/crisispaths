"""Strategy harness: execution timing, costs, history append, validation."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import crisispaths as cp
from crisispaths.harness import StrategyResult, buy_and_hold_equal_weight, run_strategy

CLOCK = cp.Clock(252, "business")
N_HIST = 5


def history(n=N_HIST, assets=("A", "B"), price=100.0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(price, index=idx, columns=list(assets))


def spike_at(pos, col="A", value=1.0):
    def strat(prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=[col])
        w.iloc[pos, 0] = value
        return w

    return strat


def cash(prices):
    return pd.DataFrame(index=prices.index, columns=[], dtype=float)


def test_weight_decided_at_bar_t_earns_bar_t_plus_1(make_path):
    rets = pd.DataFrame({"A": [0.10, 0.20, -0.10, 0.05], "B": 0.0}, index=range(4))
    res = run_strategy(spike_at(N_HIST), history(), make_path(rets), clock=CLOCK)
    assert isinstance(res, StrategyResult)
    # decided on sim bar 1 (position 5): earns bar 2 (0.20), never bar 1 (0.10)
    np.testing.assert_allclose(res.returns.to_numpy(), [0.0, np.expm1(0.20), 0.0, 0.0])
    np.testing.assert_allclose(res.equity.iloc[-1], 1.0 * (1 + np.expm1(0.20)))
    assert len(res.equity) == len(res.returns) + 1  # start point included
    assert res.equity.iloc[0] == 1.0
    assert res.weights["A"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_cash_strategy_exact_initial_capital(make_path):
    rets = pd.DataFrame(np.random.default_rng(1).normal(0, 0.05, (30, 2)), columns=["A", "B"])
    res = run_strategy(cash, history(), make_path(rets), clock=CLOCK, initial_capital=7.5,
                       costs_bps=25)
    assert (res.equity == 7.5).all()
    assert (res.returns == 0.0).all()
    assert (res.turnover == 0.0).all()
    m = cp.path_metrics(res.equity, history_peak=None)
    assert m["max_drawdown_from_start"] == 0.0 and m["max_drawdown_from_peak"] == 0.0


def test_costs_equal_bps_times_turnover(make_path):
    # weights decided at positions 4,5,6 are 0.5, 1.0, 0.0 -> executed on sim bars 1,2,3
    def strat(prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=["A"])
        w.iloc[N_HIST - 1, 0] = 0.5
        w.iloc[N_HIST, 0] = 1.0
        return w

    rets = pd.DataFrame({"A": [0.10, 0.20, 0.0, 0.0], "B": 0.0})
    res = run_strategy(strat, history(), make_path(rets), clock=CLOCK, costs_bps=10,
                       cost_multiplier=2.0)
    np.testing.assert_allclose(res.turnover.to_numpy(), [0.5, 0.5, 1.0, 0.0])
    rate = 10 / 1e4 * 2.0
    exp = [
        0.5 * np.expm1(0.10) - 0.5 * rate,
        1.0 * np.expm1(0.20) - 0.5 * rate,
        0.0 - 1.0 * rate,
        0.0,
    ]
    np.testing.assert_allclose(res.returns.to_numpy(), exp)
    np.testing.assert_allclose(res.meta["total_cost"], 2.0 * rate)


def test_shorts_cash_and_leverage_cap(make_path):
    def strat(prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=["A", "B"])
        w["A"], w["B"] = -2.0, 2.0
        return w

    rets = pd.DataFrame({"A": [0.01] * 3, "B": [0.02] * 3})
    res = run_strategy(strat, history(), make_path(rets), clock=CLOCK, leverage_cap=1.0)
    np.testing.assert_allclose(res.weights.iloc[0].to_numpy(), [-0.5, 0.5])
    np.testing.assert_allclose(res.weights.abs().sum(axis=1).to_numpy(), 1.0)
    # leftover cash earns cash_return
    half = lambda p: pd.DataFrame({"A": 0.5}, index=p.index)  # noqa: E731
    z = pd.DataFrame({"A": [0.0] * 3, "B": 0.0})
    res = run_strategy(half, history(), make_path(z), clock=CLOCK, cash_return=0.01)
    np.testing.assert_allclose(res.returns.to_numpy(), 0.5 * 0.01)


def test_ruin_is_flagged_and_never_nan(make_path):
    short = lambda p: pd.DataFrame({"A": -1.0}, index=p.index)  # noqa: E731
    rets = pd.DataFrame({"A": [0.0, 2.0, 0.1, -0.5], "B": 0.0})  # +639% on a short
    res = run_strategy(short, history(), make_path(rets), clock=CLOCK)
    assert res.meta["ruined"] is True
    assert res.meta["ruin_bar"] == 2
    assert not res.equity.isna().any() and not res.returns.isna().any()
    assert res.returns.iloc[1] == -1.0
    assert (res.equity.iloc[2:] == 0.0).all()  # absorbing
    assert res.returns.iloc[2:].eq(0.0).all()
    # an asset that goes to a total loss (log return -inf)
    long_a = lambda p: pd.DataFrame({"A": 1.0}, index=p.index)  # noqa: E731
    rets = pd.DataFrame({"A": [0.0, -np.inf, 0.3], "B": 0.0})
    res = run_strategy(long_a, history(), make_path(rets), clock=CLOCK)
    assert res.meta["ruined"] is True and res.equity.iloc[-1] == 0.0


def test_history_appended_and_excluded_from_result(make_path):
    n_hist = 60
    idx = pd.bdate_range("2019-01-01", periods=n_hist)
    h = pd.DataFrame({"A": 100.0 * np.exp(0.002 * np.arange(n_hist)), "B": 50.0}, index=idx)
    seen = {}

    def sma(prices):
        seen["rows"] = len(prices)
        seen["last_hist_row"] = prices.iloc[n_hist - 1].copy()
        sig = (prices > prices.rolling(50).mean()).astype(float)
        sig[prices.rolling(50).mean().isna()] = np.nan
        return sig

    rets = pd.DataFrame({"A": [0.001] * 10, "B": 0.0})
    res = run_strategy(sma, h, make_path(rets), clock=CLOCK)
    assert seen["rows"] == n_hist + 10
    np.testing.assert_allclose(seen["last_hist_row"].to_numpy(), h.iloc[-1].to_numpy())
    # decided on the last history bar (SMA warm), executed on the first simulated bar
    assert res.weights["A"].iloc[0] == 1.0
    assert res.meta["nan_weight_bars"] == 0
    # result covers only the simulated segment (+ the start point)
    assert len(res.returns) == 10 and len(res.equity) == 11
    assert res.equity.index[0] == h.index[-1]
    assert res.returns.index[0] > h.index[-1]
    # path prices compound from the last history close
    np.testing.assert_allclose(res.meta["last_close"]["A"], h["A"].iloc[-1])


def test_history_peak_in_equity_units(make_path):
    idx = pd.bdate_range("2019-01-01", periods=4)
    h = pd.DataFrame({"A": [100.0, 120.0, 90.0, 99.0]}, index=idx)
    long_a = lambda p: pd.DataFrame({"A": 1.0}, index=p.index)  # noqa: E731
    res = run_strategy(long_a, h, make_path(pd.DataFrame({"A": [0.0, 0.0]})), clock=CLOCK,
                       initial_capital=2.0)
    # history equity under the strategy (held from bar 1): 1, 1.2, 0.9, 0.99 -> normalised to 2
    np.testing.assert_allclose(res.meta["history_peak"], 2.0 * 1.2 / 0.99)


@pytest.mark.parametrize(
    "bad, msg",
    [
        (lambda p: pd.DataFrame({"Z": 1.0}, index=p.index), "not in assets"),
        (lambda p: pd.DataFrame({"A": np.inf}, index=p.index), "non-finite"),
        (lambda p: pd.DataFrame({"A": 1.0}, index=p.index[1:]), "index"),
        (lambda p: pd.Series(1.0, index=p.index), "mode"),
    ],
)
def test_validation_errors(bad, msg, make_path):
    rets = pd.DataFrame({"A": [0.0, 0.0], "B": 0.0})
    with pytest.raises(ValueError, match=msg):
        run_strategy(bad, history(), make_path(rets), clock=CLOCK)


def test_validation_of_arguments(make_path):
    rets = pd.DataFrame({"A": [0.0, 0.0], "B": 0.0})
    p = make_path(rets)
    with pytest.raises(ValueError, match="mode"):
        run_strategy(cash, history(), p, clock=CLOCK, mode="nope")
    with pytest.raises(ValueError, match="history_prices"):
        run_strategy(cash, history(assets=("A",)), p, clock=CLOCK)
    with pytest.raises(ValueError, match="costs_bps"):
        run_strategy(cash, history(), p, clock=CLOCK, costs_bps=-1)


def test_returns_mode_user_owns_timing(make_path):
    rets = pd.DataFrame({"A": [0.10, 0.20, -0.10], "B": 0.0})
    p = make_path(rets)

    def own(prices):  # already-earned strategy returns, aligned to bars
        return pd.Series(0.01, index=prices.index)

    res = run_strategy(own, history(), p, clock=CLOCK, mode="returns")
    np.testing.assert_allclose(res.returns.to_numpy(), 0.01)
    np.testing.assert_allclose(res.equity.iloc[-1], 1.01**3)
    assert res.weights.shape[1] == 0
    with pytest.raises(ValueError, match="returns"):
        run_strategy(own, history(), p, clock=CLOCK, mode="returns", costs_bps=5)


def test_buy_and_hold_equal_weight_benchmark(make_path):
    rets = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.10]})
    res = run_strategy(buy_and_hold_equal_weight, history(), make_path(rets), clock=CLOCK)
    np.testing.assert_allclose(res.weights.to_numpy(), 0.5)
    np.testing.assert_allclose(res.returns.iloc[0], 0.5 * np.expm1(0.10))


# --- properties -----------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.integers(5, 60))
def test_cash_has_zero_drawdown(seed, n):
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0, 0.1, (n, 2)), columns=["A", "B"])
    p = cp.GeneratedPath(rets, pd.DataFrame(index=rets.index), np.zeros(n, int), np.ones(n), {})
    res = run_strategy(cash, history(), p, clock=CLOCK)
    m = cp.path_metrics(res.equity, history_peak=res.meta["history_peak"])
    assert m["max_drawdown_from_peak"] == 0.0 and m["max_drawdown_from_start"] == 0.0


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_ruin_probability_monotone_in_crash_depth(seed):
    rng = np.random.default_rng(seed)
    n_paths, n = 40, 80
    base = rng.normal(-0.002, 0.03, (n_paths, n))
    long_a = lambda p: pd.DataFrame({"A": 1.0}, index=p.index)  # noqa: E731
    p_ruin = []
    for depth in (0.5, 1.0, 1.5, 2.0, 3.0):  # scale the whole path: deeper crash
        ruined = 0
        for k in range(n_paths):
            df = pd.DataFrame({"A": depth * base[k], "B": 0.0})
            path = cp.GeneratedPath(df, pd.DataFrame(index=df.index), np.zeros(n, int),
                                    np.ones(n), {})
            res = run_strategy(long_a, history(), path, clock=CLOCK)
            m = cp.path_metrics(res.equity, history_peak=None, ruin_level=0.7)
            ruined += bool(m["ruined"])
        p_ruin.append(ruined / n_paths)
    assert all(a <= b + 1e-12 for a, b in zip(p_ruin, p_ruin[1:], strict=False)), p_ruin


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.0, 3.0), st.floats(0.0, 200.0))
def test_equity_never_negative_long_only_capped(seed, scale, bps):
    rng = np.random.default_rng(seed)
    n = 40
    rets = pd.DataFrame(rng.normal(0, 0.5, (n, 2)), columns=["A", "B"])  # brutal returns
    path = cp.GeneratedPath(rets, pd.DataFrame(index=rets.index), np.zeros(n, int),
                            np.ones(n), {})
    w_raw = rng.uniform(0, 1, (N_HIST + n, 2)) * scale * 5

    def strat(prices):
        return pd.DataFrame(w_raw, index=prices.index, columns=["A", "B"])

    res = run_strategy(strat, history(), path, clock=CLOCK, leverage_cap=1.0, costs_bps=bps)
    assert (res.equity >= 0).all()
    assert np.isfinite(res.equity.to_numpy()).all()

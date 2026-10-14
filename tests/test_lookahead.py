"""Look-ahead detector: truncated runs must reproduce the full run on the shared prefix."""

import numpy as np
import pandas as pd
import pytest

import crisispaths as cp
from crisispaths.harness import run_strategy
from crisispaths.lookahead import LookaheadError, LookaheadReport, check_lookahead

CLOCK = cp.Clock(252, "business")


def prices(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)), index=idx, columns=["A", "B"]
    )


def sma_trend(p):
    ma = p.rolling(20).mean()
    return (p > ma).astype(float).where(ma.notna()) / p.shape[1]


def leaky(p):
    return (p.shift(-1) > p).astype(float) / p.shape[1]


def test_clean_strategy_passes():
    rep = check_lookahead(sma_trend, prices())
    assert isinstance(rep, LookaheadReport)
    assert rep.passed and rep.offending_bars == []
    assert len(rep.cut_points) == 3


def test_planted_leak_is_caught_and_bars_listed():
    p = prices()
    rep = check_lookahead(leaky, p)
    assert not rep.passed
    assert rep.offending_bars
    assert set(rep.offending_bars) <= set(p.index)
    # the offending bar is the last bar of a prefix (it peeked one bar ahead)
    cuts = {p.index[k - 1] for k in rep.cut_points}
    assert set(rep.offending_bars) & cuts


def test_invalid_cut_points():
    with pytest.raises(ValueError, match="cut_points"):
        check_lookahead(sma_trend, prices(), cut_points=(0.0, 1.5))


def test_harness_raises_or_warns(make_path):
    h = prices()
    rets = pd.DataFrame({"A": [0.0] * 5, "B": 0.0})
    path = make_path(rets)
    with pytest.raises(LookaheadError, match="look-ahead"):
        run_strategy(leaky, h, path, clock=CLOCK, check_lookahead=True)
    with pytest.warns(UserWarning, match="look-ahead"):
        res = run_strategy(leaky, h, path, clock=CLOCK, check_lookahead=True,
                           on_lookahead="warn")
    assert res.meta["lookahead"]["passed"] is False
    ok = run_strategy(sma_trend, h, path, clock=CLOCK, check_lookahead=True)
    assert ok.meta["lookahead"]["passed"] is True
    with pytest.raises(ValueError, match="on_lookahead"):
        run_strategy(sma_trend, h, path, clock=CLOCK, check_lookahead=True, on_lookahead="x")


def test_returns_mode_bypass_is_documented_and_not_checked(make_path):
    """Returns mode cannot be audited: a leaky returns strategy is accepted (with a warning)."""
    h = prices()
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, (50, 2)), columns=["A", "B"])
    path = make_path(rets)

    def peeking(frame):  # oracle: books the size of the NEXT bar's move, look-ahead by design
        return frame["A"].pct_change().shift(-1).abs().fillna(0.0)

    with pytest.warns(UserWarning, match="returns mode"):
        res = run_strategy(peeking, h, path, clock=CLOCK, mode="returns", check_lookahead=True)
    assert res.meta["lookahead"] is None
    honest = run_strategy(
        lambda f: pd.DataFrame({"A": 1.0}, index=f.index), h, path, clock=CLOCK
    )
    assert res.equity.iloc[-1] > honest.equity.iloc[-1]  # the leak pays; nothing flags it

"""Strategy harness: run a strategy on a generated path.

Contract (``mode="weights"``)
-----------------------------
``strategy(prices) -> weights``: ``prices`` is one DataFrame (history followed
by the simulated path, one column per asset); ``weights`` has the same index
and columns that are a subset of the assets. NaN means zero, leftover is cash.

Timing: the weights set at bar ``t`` from data through the close of ``t`` earn
the return of bar ``t + 1``, never bar ``t`` (a shift of exactly one bar). The
weights of the last history bar therefore earn the first simulated bar.

Prices: the path's asset log returns are compounded from the LAST CLOSE of the
user's history and appended to it, so rolling indicators are warm from the
first simulated bar. Everything returned covers only the simulated segment.

Accounting: weights are rebalanced to target every bar (turnover is the sum of
absolute weight changes, drift ignored). Positions carry over from the
history: the strategy is assumed to have been running through it, so there is
no artificial entry cost at the segment start. Cost per bar is
``costs_bps / 1e4 * cost_multiplier * turnover``. Cash (and borrowing, when
gross exposure exceeds one) earns ``cash_return`` per bar. Portfolio returns
are simple returns; a bar return at or below -100% is clipped at -1, ruin is
flagged and equity stays at zero afterwards.

``mode="returns"``: ``strategy(prices) -> Series`` of strategy returns per bar.
The user owns timing and look-ahead there: the harness applies no shift, no
costs and no leverage cap, and cannot audit the series.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clock import Clock, session_slots
from .generate import GeneratedPath
from .lookahead import LookaheadError
from .lookahead import check_lookahead as _audit

Strategy = Callable[[pd.DataFrame], pd.DataFrame]
ReturnsStrategy = Callable[[pd.DataFrame], pd.Series]

MODES = ("weights", "returns")
LOOKAHEAD_ACTIONS = ("raise", "warn")


@dataclass
class StrategyResult:
    """Simulated-segment result.

    ``equity`` has ``n + 1`` points: the start (``initial_capital``, indexed at
    the last history bar) then one per simulated bar. ``returns``, ``weights``
    (executed, i.e. held during the bar) and ``turnover`` have ``n`` rows.
    """

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    meta: dict = field(default_factory=dict)


def buy_and_hold_equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: equal weight in every asset, held (rebalanced to 1/N) every bar."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def _intraday_index(history_index: pd.DatetimeIndex, n: int, clock: Clock) -> pd.Index | None:
    """Session grid continued after the history: same times of day, next session dates."""
    tail = history_index[-int(60 * float(clock.bars_per_day)):]
    slots = session_slots(tail)
    if len(slots) < 2:
        return None
    n_days = -(-n // len(slots))
    start = history_index[-1].normalize() + pd.Timedelta(days=1)
    if clock.calendar == "business":
        days = pd.bdate_range(start, periods=n_days)
    else:
        days = pd.date_range(start, periods=n_days, freq="D")
    offs = pd.to_timedelta(np.asarray(slots), unit="s")
    return pd.DatetimeIndex((days.to_numpy()[:, None] + offs.to_numpy()[None, :]).ravel()[:n])


def _simulated_index(history_index: pd.Index, n: int, clock: Clock) -> pd.Index:
    if clock.is_intraday and isinstance(history_index, pd.DatetimeIndex):
        idx = _intraday_index(history_index, n, clock)
        if idx is not None:
            return idx
    last = history_index[-1]
    if not isinstance(history_index, pd.DatetimeIndex):
        if np.issubdtype(history_index.dtype, np.integer):
            return pd.RangeIndex(int(last) + 1, int(last) + 1 + n)
        raise ValueError("history_prices needs a DatetimeIndex or an integer index")
    if len(history_index) >= 2:
        step = pd.Series(history_index).diff().dropna().median()
    else:
        step = pd.Timedelta(days=365.25 / float(clock.bars_per_year))
    if pd.Timedelta(hours=20) <= step <= pd.Timedelta(hours=28):
        start = last + pd.Timedelta(days=1)
        if clock.calendar == "business":
            return pd.bdate_range(start, periods=n)
        return pd.date_range(start, periods=n, freq="D")
    return pd.DatetimeIndex([last + step * (k + 1) for k in range(n)])


def _check_args(mode, costs_bps, cost_multiplier, initial_capital, leverage_cap, cash_return,
                on_lookahead):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if on_lookahead not in LOOKAHEAD_ACTIONS:
        raise ValueError(f"on_lookahead must be one of {LOOKAHEAD_ACTIONS}, got {on_lookahead!r}")
    if not np.isfinite(costs_bps) or costs_bps < 0:
        raise ValueError(f"costs_bps must be a finite number >= 0, got {costs_bps}")
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError(f"cost_multiplier must be a finite number >= 0, got {cost_multiplier}")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    if leverage_cap is not None and not (np.isfinite(leverage_cap) and leverage_cap > 0):
        raise ValueError(f"leverage_cap must be > 0 or None, got {leverage_cap}")
    if not np.isfinite(cash_return) or cash_return <= -1:
        raise ValueError(f"cash_return must be a finite per-bar return > -1, got {cash_return}")
    if mode == "returns" and (costs_bps > 0 or leverage_cap is not None):
        raise ValueError(
            "costs_bps and leverage_cap need weights; in mode='returns' the strategy owns its "
            "returns net of costs"
        )


def _clean_weights(raw, index: pd.Index, assets: list[str]) -> pd.DataFrame:
    if isinstance(raw, pd.Series):
        raise ValueError(
            "the strategy returned a Series; weights mode expects a DataFrame of weights "
            "(use mode='returns' for a Series of strategy returns)"
        )
    if not isinstance(raw, pd.DataFrame):
        raise ValueError(f"the strategy must return a DataFrame, got {type(raw).__name__}")
    if raw.columns.has_duplicates:
        raise ValueError("weights has duplicate columns")
    unknown = [c for c in raw.columns if c not in assets]
    if unknown:
        raise ValueError(f"weights columns {unknown} are not in assets {assets}")
    if not raw.index.equals(index):
        raise ValueError(
            f"weights index does not match the price index (got {len(raw)} rows, "
            f"expected {len(index)} with the same labels)"
        )
    w = raw.reindex(columns=assets).astype(float)
    if np.isinf(w.to_numpy()).any():
        raise ValueError("weights contain non-finite values (inf); NaN is allowed and means 0")
    return w


def _segment(w_exec: np.ndarray, simple: np.ndarray, cost_rate: float, cash_return: float,
             w_before: np.ndarray):
    """Per-bar portfolio return, turnover, cost for a block of executed weights."""
    prev = np.vstack([w_before[None, :], w_exec[:-1]])
    turnover = np.abs(w_exec - prev).sum(axis=1)
    cost = cost_rate * turnover
    gross = (w_exec * simple).sum(axis=1) + (1.0 - w_exec.sum(axis=1)) * cash_return
    return gross - cost, turnover, cost


def _absorb(ret: np.ndarray):
    """Clip at -1; zero everything after the first total loss. Returns (ret, ruin_pos|None)."""
    ret = np.maximum(ret, -1.0)
    hit = np.where(ret <= -1.0)[0]
    pos = int(hit[0]) if hit.size else None
    if pos is not None:
        ret[pos + 1:] = 0.0
    return ret, pos


def run_strategy(
    strategy,
    history_prices: pd.DataFrame,
    path: GeneratedPath,
    *,
    clock: Clock,
    costs_bps: float = 0.0,
    cost_multiplier: float = 1.0,
    mode: str = "weights",
    initial_capital: float = 1.0,
    leverage_cap: float | None = None,
    cash_return: float = 0.0,
    check_lookahead: bool = False,
    on_lookahead: str = "raise",
) -> StrategyResult:
    """Run ``strategy`` on ``history_prices`` followed by the simulated ``path``.

    ``check_lookahead`` replays the strategy once on the history only (see
    :mod:`crisispaths.lookahead`); ``on_lookahead`` is ``"raise"`` or ``"warn"``.
    Weights mode only: returns mode cannot be audited and warns instead.
    """
    _check_args(mode, costs_bps, cost_multiplier, initial_capital, leverage_cap, cash_return,
                on_lookahead)
    assets = list(path.returns.columns)
    missing = [a for a in assets if a not in history_prices.columns]
    if missing:
        raise ValueError(f"history_prices is missing path asset(s) {missing}")
    hist = history_prices[assets].astype(float)
    if len(hist) == 0:
        raise ValueError("history_prices is empty")
    last = hist.iloc[-1]
    if not (np.isfinite(last.to_numpy()).all() and (last > 0).all()):
        raise ValueError("the last history close must be finite and positive for every asset")
    if (hist.to_numpy() <= 0).any():
        raise ValueError("history_prices must be positive")
    if not hist.index.is_monotonic_increasing or hist.index.has_duplicates:
        raise ValueError("history_prices index must be increasing and unique")

    logret = path.returns.to_numpy(float)
    if np.isnan(logret).any() or np.isposinf(logret).any():
        raise ValueError("path returns must not contain NaN or +inf (-inf is a total loss)")
    n, n_hist = logret.shape[0], len(hist)
    sim_index = _simulated_index(hist.index, n, clock)
    with np.errstate(over="ignore"):
        sim_px = last.to_numpy()[None, :] * np.exp(np.cumsum(logret, axis=0))
    frame = pd.concat([hist, pd.DataFrame(sim_px, index=sim_index, columns=assets)])
    frame.index.name = hist.index.name

    lookahead = None
    if check_lookahead:
        if mode == "returns":
            warnings.warn(
                "look-ahead check skipped: returns mode cannot be audited, the strategy "
                "owns its timing",
                UserWarning,
                stacklevel=2,
            )
        else:
            rep = _audit(strategy, hist)
            lookahead = rep.summary()
            if not rep.passed:
                if on_lookahead == "raise":
                    raise LookaheadError(rep.message())
                warnings.warn(rep.message(), UserWarning, stacklevel=2)

    cost_rate = costs_bps / 1e4 * cost_multiplier
    simple_hist = hist.pct_change().fillna(0.0).to_numpy()
    simple_sim = np.expm1(logret)
    simple_sim = np.maximum(simple_sim, -1.0)

    meta: dict = {
        "mode": mode,
        "history_bars": n_hist,
        "sim_bars": n,
        "last_close": last.copy(),
        "price_basis": "last history close compounded with path log returns",
        "costs_bps": float(costs_bps),
        "cost_multiplier": float(cost_multiplier),
        "leverage_cap": leverage_cap,
        "cash_return": float(cash_return),
        "bars_per_year": float(clock.bars_per_year),
        "lookahead": lookahead,
    }

    if mode == "returns":
        raw = strategy(frame.copy())
        if not isinstance(raw, pd.Series):
            raise ValueError("in mode='returns' the strategy must return a Series of returns")
        if not raw.index.equals(frame.index):
            raise ValueError("returns index does not match the price index")
        r = raw.astype(float).fillna(0.0).to_numpy()
        if np.isinf(r).any():
            raise ValueError("strategy returns contain non-finite values (inf)")
        ret_h, _ = _absorb(r[:n_hist].copy())
        ret, ruin_pos = _absorb(r[n_hist:].copy())
        weights = pd.DataFrame(index=sim_index, columns=[], dtype=float)
        turnover = pd.Series(np.nan, index=sim_index, name="turnover")
        cost = np.zeros(n)
        meta["nan_weight_bars"] = 0
    else:
        raw = strategy(frame.copy())
        w = _clean_weights(raw, frame.index, assets)
        # executed bars of the simulated segment whose decision row had a NaN weight
        n_nan = int(raw.iloc[n_hist - 1:-1].isna().any(axis=1).sum()) if raw.shape[1] else 0
        raw_w = w.fillna(0.0).to_numpy()
        gross = np.abs(raw_w).sum(axis=1)
        capped = 0
        if leverage_cap is not None:
            over = gross > leverage_cap
            scale = np.ones_like(gross)
            scale[over] = leverage_cap / gross[over]
            capped = int(over.sum())
            raw_w = raw_w * scale[:, None]
        w_exec = np.vstack([np.zeros((1, len(assets))), raw_w[:-1]])  # decided t, earned t+1
        ret_h_raw, _, _ = _segment(w_exec[:n_hist], simple_hist, cost_rate, cash_return,
                                   np.zeros(len(assets)))
        ret_h, _ = _absorb(ret_h_raw)
        ret_raw, turn, cost = _segment(w_exec[n_hist:], simple_sim, cost_rate, cash_return,
                                       w_exec[n_hist - 1])
        ret, ruin_pos = _absorb(ret_raw)
        if ruin_pos is not None:
            turn[ruin_pos + 1:] = 0.0
            cost[ruin_pos + 1:] = 0.0
        weights = pd.DataFrame(w_exec[n_hist:], index=sim_index, columns=assets)
        turnover = pd.Series(turn, index=sim_index, name="turnover")
        meta["nan_weight_bars"] = n_nan
        meta["capped_bars"] = capped

    h_eq = np.cumprod(1.0 + ret_h)
    peak = initial_capital
    if h_eq[-1] > 0:
        peak = max(initial_capital, float(initial_capital * h_eq.max() / h_eq[-1]))
    equity = np.concatenate([[initial_capital], initial_capital * np.cumprod(1.0 + ret)])
    eq_index = pd.Index([hist.index[-1], *sim_index])
    meta.update(
        history_peak=float(peak),
        ruined=ruin_pos is not None,
        ruin_bar=None if ruin_pos is None else ruin_pos + 1,
        total_cost=float(np.sum(cost)),
    )
    return StrategyResult(
        equity=pd.Series(equity, index=eq_index, name="equity"),
        returns=pd.Series(ret, index=sim_index, name="returns"),
        weights=weights,
        turnover=turnover,
        meta=meta,
    )


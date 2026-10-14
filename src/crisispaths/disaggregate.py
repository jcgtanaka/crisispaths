"""Intraday disaggregation: simulate daily, then split each day into bars.

Design
------
Paths are ALWAYS simulated at the calibration frequency (daily). The user's
intraday bar clock is reached by *disaggregation*: every simulated daily log
return is split into the bars of its session by a Brownian bridge, so the bars
of a session sum EXACTLY to the daily return. Minute bars are never simulated
directly: summing many small independent shocks would wash out the fat daily
crash tails (central limit theorem) and the memory would explode.

Bridge. With ``S`` bars of variance shares ``w_k`` (sum 1), unconditioned
innovations ``x_k = sigma * sqrt(w_k) * z_k`` and the daily return ``d``, the
bars are ``r_k = x_k + w_k * (d - sum_j x_j)``. Then ``sum_k r_k = d`` and the
conditional variance of bar ``k`` is proportional to ``w_k (1 - w_k)``.

* ``z`` are Student-t with ``df`` degrees of freedom (scaled to unit variance),
  ``df`` estimated from the user's intraday data (default 5).
* ``sigma`` is ``max(|d|, daily_vol)`` per asset and day: the intraday noise
  scales with the size of the day's move and never falls below the asset's
  calm daily volatility.
* Cross-asset dependence inside the day comes from the DAILY residual
  correlation matrix (``corr``). The Epps effect (correlations shrink at high
  frequency) is not modelled, so within-day co-movement of fast bars is
  overstated.

Profile. :func:`estimate_intraday_profile` measures, per slot of the session,
the share of the session variance (mean squared bar return, normalised to sum
1). The first bar of a session contains the overnight gap when the calendar has
one, so slot 0 is inflated; ``overnight_share`` is the excess of slot 0 over
its linear extrapolation from slots 1 and 2. Per-asset shares are shrunk toward
the pooled shares when an asset has fewer than ``min_sessions`` sessions
(weight ``n / min_sessions`` on the asset's own estimate).

Coarser bars (weekly, monthly): :func:`aggregate_log_returns` sums daily log
returns in blocks.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clock import Clock, session_slots

DEFAULT_DF = 5.0
DF_BOUNDS = (4.5, 50.0)
DEFAULT_MIN_SESSIONS = 120
CHUNK_SESSIONS = 64  # sessions disaggregated at a time: bounds the temporaries


@dataclass(frozen=True)
class IntradayProfile:
    """Per-slot variance shares of a session, per asset (``weights``) and pooled.

    ``weights`` rows sum to 1 and are what :func:`disaggregate` uses.
    ``raw_weights`` are the unshrunk per-asset estimates (equal to ``weights``
    for a flat profile). ``slot_times`` are seconds after midnight (empty when
    the profile is flat and no slot grid is known).
    """

    assets: tuple[str, ...]
    slots: int
    weights: np.ndarray
    pooled: np.ndarray
    raw_weights: np.ndarray
    n_sessions: np.ndarray
    shrinkage: np.ndarray
    overnight_share: np.ndarray
    has_overnight_gap: bool
    df: float
    daily_vol: np.ndarray
    source: str  # "estimated" | "flat"
    slot_times: tuple[int, ...] = field(default=())

    @classmethod
    def flat(cls, slots: int, assets: Sequence[str], daily_vol=None, *, df: float = DEFAULT_DF,
             has_overnight_gap: bool = True) -> IntradayProfile:
        """Uniform profile (every bar the same variance share)."""
        if slots < 1:
            raise ValueError("slots must be >= 1")
        a = len(assets)
        w = np.full((a, slots), 1.0 / slots)
        vol = np.full(a, 0.01) if daily_vol is None else np.asarray(daily_vol, float)
        return cls(
            assets=tuple(map(str, assets)), slots=int(slots), weights=w, pooled=w[0].copy(),
            raw_weights=w.copy(), n_sessions=np.zeros(a), shrinkage=np.ones(a),
            overnight_share=np.full(a, np.nan), has_overnight_gap=has_overnight_gap,
            df=float(df), daily_vol=vol, source="flat",
        )


# --- estimation --------------------------------------------------------------------------------


def _grid(prices: pd.DataFrame, slots: list[int]):
    """Log prices on a (sessions, slots, assets) grid; NaN where a bar is missing."""
    idx = pd.DatetimeIndex(prices.index)
    sess = idx.normalize()
    sec = (idx.hour * 3600 + idx.minute * 60 + idx.second).to_numpy()
    dates, d_inv = np.unique(sess.to_numpy(), return_inverse=True)
    s_arr = np.asarray(slots)
    pos = np.searchsorted(s_arr, sec)
    pos = np.clip(pos, 0, len(s_arr) - 1)
    ok = s_arr[pos] == sec
    grid = np.full((len(dates), len(slots), prices.shape[1]), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(prices.to_numpy(float))
    grid[d_inv[ok], pos[ok]] = lp[ok]
    return dates, grid


def _bar_returns(grid: np.ndarray) -> np.ndarray:
    """Bar log returns (sessions, slots, assets); the first bar spans the overnight gap."""
    n, s, a = grid.shape
    flat = grid.reshape(n * s, a)
    r = np.full_like(flat, np.nan)
    r[1:] = flat[1:] - flat[:-1]
    return r.reshape(n, s, a)


def aggregate_to_daily(prices_intraday: pd.DataFrame, clock: Clock | None = None) -> pd.DataFrame:
    """Daily prices: the last valid price of each session date (sum of its bar log returns)."""
    idx = pd.DatetimeIndex(prices_intraday.index)
    out = prices_intraday.groupby(idx.normalize()).last()
    out.index.name = prices_intraday.index.name
    return out


def estimate_intraday_profile(
    prices_intraday: pd.DataFrame,
    clock: Clock,
    *,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    default_df: float = DEFAULT_DF,
) -> IntradayProfile:
    """Estimate the per-slot variance profile of the session from intraday prices.

    Slots are the times of day present in at least half of the sessions; a bar
    that is missing is skipped (returns touching it are NaN), never zero-filled.
    """
    if not isinstance(prices_intraday.index, pd.DatetimeIndex):
        raise ValueError("intraday prices need a DatetimeIndex")
    slots = session_slots(prices_intraday.index)
    if len(slots) < 2:
        raise ValueError("fewer than two bar times per session: the data is not intraday")
    dates, grid = _grid(prices_intraday, slots)
    r = _bar_returns(grid)  # (D, S, A)
    n_d, s, a = r.shape
    valid = np.isfinite(r)
    cnt = valid.sum(axis=0)  # (S, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(cnt >= 5, np.nansum(np.where(valid, r, 0.0) ** 2, axis=0) / cnt, np.nan)
    tot = np.nansum(var, axis=0)
    raw = np.where(np.isfinite(var), var, np.nan) / np.where(tot > 0, tot, np.nan)  # (S, A)
    raw = raw.T  # (A, S)
    n_sess = (valid.sum(axis=1) >= max(1, s // 2)).sum(axis=0).astype(float)  # (A,)

    have = np.isfinite(raw).all(axis=1) & (n_sess > 0)
    if not have.any():
        raise ValueError("no asset has enough intraday bars to estimate a profile")
    wts = np.where(have, n_sess, 0.0)
    pooled = np.nansum(np.where(have[:, None], raw, 0.0) * wts[:, None], axis=0) / wts.sum()
    pooled = pooled / pooled.sum()
    lam = np.clip(n_sess / float(min_sessions), 0.0, 1.0)
    lam = np.where(have, lam, 0.0)
    raw_f = np.where(have[:, None], raw, pooled[None, :])
    w = lam[:, None] * raw_f + (1.0 - lam[:, None]) * pooled[None, :]
    w = w / w.sum(axis=1, keepdims=True)

    gap = bool(clock.overnight_gap)
    over = np.full(a, np.nan)
    if gap:
        if s >= 3:
            ref = np.maximum(2.0 * w[:, 1] - w[:, 2], np.minimum(w[:, 1], w[:, 2]))
        else:
            ref = w[:, 1]
        over = np.clip(w[:, 0] - ref, 0.0, None)

    sd = np.sqrt(np.where(np.isfinite(var), var, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / sd[None, :, :]
    zz = z[np.isfinite(z)]
    df = float(default_df)
    if zz.size >= 2000:
        m2 = float(np.mean(zz**2))
        kurt = float(np.mean(zz**4)) / (m2 * m2) - 3.0
        if np.isfinite(kurt):
            df = 4.0 + 6.0 / kurt if kurt > 0.05 else DF_BOUNDS[1]
            df = float(np.clip(df, *DF_BOUNDS))

    daily = np.log(aggregate_to_daily(prices_intraday)).diff().iloc[1:]
    dvol = daily.std().to_numpy(float)
    dvol = np.where(np.isfinite(dvol) & (dvol > 0), dvol, 0.01)
    return IntradayProfile(
        assets=tuple(map(str, prices_intraday.columns)), slots=s, weights=w, pooled=pooled,
        raw_weights=raw_f, n_sessions=n_sess, shrinkage=np.where(have, lam, 0.0),
        overnight_share=over, has_overnight_gap=gap, df=df, daily_vol=dvol,
        source="estimated", slot_times=tuple(int(x) for x in slots),
    )


# --- generation --------------------------------------------------------------------------------


def disaggregate(
    daily_log_returns,
    profile: IntradayProfile,
    rng: np.random.Generator,
    *,
    corr: np.ndarray | None = None,
    df: float | None = None,
    chunk_sessions: int = CHUNK_SESSIONS,
) -> np.ndarray:
    """Split daily log returns into bars with a Brownian bridge.

    ``daily_log_returns`` is (n_days, n_assets) in the profile's asset order.
    Returns (n_days * slots, n_assets); the bars of each session sum to the
    daily return to rounding error (1e-12). ``corr`` is the cross-asset
    correlation of the innovations (identity when omitted). Draws are
    sequential from ``rng``, so a prefix of the days is identical whatever the
    total length.
    """
    d = np.asarray(daily_log_returns, dtype=float)
    if d.ndim != 2:
        raise ValueError("daily_log_returns must be a 2-D (days x assets) array")
    n, a = d.shape
    s = profile.slots
    if a != len(profile.assets):
        raise ValueError(
            f"{a} asset columns but the profile has {len(profile.assets)} assets"
        )
    nu = float(profile.df if df is None else df)
    if not nu > 2:
        raise ValueError("df must be > 2")
    w = profile.weights / profile.weights.sum(axis=1, keepdims=True)  # (A, S)
    wT = w.T  # (S, A)
    root = np.sqrt(wT)[None, :, :]
    vol = np.asarray(profile.daily_vol, float)
    if corr is None:
        chol = None
    else:
        c = np.asarray(corr, float)
        if c.shape != (a, a):
            raise ValueError(f"corr must be {a}x{a}")
        chol = np.linalg.cholesky(c + 1e-10 * np.eye(a))
    unit = np.sqrt((nu - 2.0) / nu)
    out = np.empty((n * s, a))
    for lo in range(0, n, max(1, int(chunk_sessions))):
        hi = min(n, lo + chunk_sessions)
        dd = d[lo:hi]
        z = rng.standard_t(nu, size=(hi - lo, s, a)) * unit
        if chol is not None:
            z = z @ chol.T
        sig = np.maximum(np.abs(dd), vol[None, :])
        x = z * root * sig[:, None, :]
        adj = dd - x.sum(axis=1)
        x += wT[None, :, :] * adj[:, None, :]
        out[lo * s: hi * s] = x.reshape((hi - lo) * s, a)
    return out


def aggregate_log_returns(returns, k: int) -> np.ndarray:
    """Sum ``k`` consecutive log-return rows; a trailing partial block is dropped."""
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    r = np.asarray(returns, dtype=float)
    n = (r.shape[0] // int(k)) * int(k)
    return r[:n].reshape(n // int(k), int(k), *r.shape[1:]).sum(axis=1)


def warn_flat(slots: int) -> None:
    warnings.warn(
        f"no intraday history was supplied: using a flat intraday profile over {slots} bars "
        "per session (every bar gets the same variance share). Pass intraday prices to "
        "calibrate() to estimate the profile.",
        UserWarning,
        stacklevel=3,
    )

"""Synthetic intraday data with a known profile (bars of a business-day session)."""

import numpy as np
import pandas as pd


def u_shape(slots: int, depth: float = 3.0, gap_extra: float = 0.0) -> np.ndarray:
    """Variance shares, U-shaped over the session; ``gap_extra`` is added to the first slot."""
    k = np.arange(slots)
    c = (slots - 1) / 2.0
    v = 1.0 + depth * ((k - c) / c) ** 2
    v[0] += gap_extra
    return v / v.sum()


def session_index(dates, slots: int, start: str = "09:30:00", step: str = "30min"):
    offs = pd.to_timedelta(start) + pd.to_timedelta(step) * np.arange(slots)
    return pd.DatetimeIndex(
        [d + o for d in pd.DatetimeIndex(dates).normalize() for o in offs]
    )


def bridge_bars(daily: np.ndarray, shares: np.ndarray, sigma: np.ndarray, rng) -> np.ndarray:
    """Bar log returns (days * slots, assets) that sum exactly to ``daily`` per day."""
    n, a = daily.shape
    s = len(shares)
    x = rng.normal(size=(n, s, a)) * np.sqrt(shares)[None, :, None] * sigma[None, None, :]
    adj = daily - x.sum(axis=1)
    return (x + shares[None, :, None] * adj[:, None, :]).reshape(n * s, a)


def intraday_prices(daily_logret: pd.DataFrame, slots: int = 13, shares=None, seed: int = 0):
    shares = u_shape(slots) if shares is None else np.asarray(shares)
    rng = np.random.default_rng(seed)
    sigma = daily_logret.std().to_numpy()
    bars = bridge_bars(daily_logret.to_numpy(), shares, sigma, rng)
    idx = session_index(daily_logret.index, len(shares))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(bars, axis=0)), index=idx,
                        columns=daily_logret.columns)

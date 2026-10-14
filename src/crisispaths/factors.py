"""Factor proxy registry and factor return construction.

All factors are free-data proxies (Yahoo Finance and FRED). They are
approximations of the economic factor named, not the factor itself. In
particular:

* ``EQ`` is the S&P 500 price index (no dividends).
* ``TECH`` is the Nasdaq Composite return orthogonalised to ``EQ`` within each
  calibration window (a residual, so it is uncorrelated with ``EQ`` by design).
* ``USD`` is the FRED broad dollar index (DTWEXBGS, starts 2006) spliced to the
  ICE dollar index (DX-Y.NYB) before 2006.
* ``RATES`` is the return on a constant-duration (D = 8) Treasury position,
  ``-D * delta(DGS10)``.
* ``CREDIT`` is the change in the Moody's Baa minus 10-year Treasury spread
  (BAA10Y). ICE BofA high-yield OAS is deliberately not used (licensing and
  history restrictions on FRED).
* ``VOL`` is the change in log VIX.
* ``HAVEN`` is front-month gold futures (GC=F). The free series has a gap of
  roughly five months starting 2000-03; gaps are filled with the conditional
  mean given the other factors and flagged in ``DataFrame.attrs``.
* ``COMMODITY`` (optional) is WTI crude (DCOILWTICO).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clock import Clock
from .data import FactorProxySource

RATES_DURATION = 8.0
MAX_GAP_BARS = 10
MAX_GAP_DAYS = 10
_PAD_DAYS = 14

Levels = dict[str, pd.Series]


@dataclass(frozen=True)
class FactorDef:
    name: str
    inputs: tuple[str, ...]
    note: str
    build: Callable[[Levels, Sequence[tuple[str, str]] | None], pd.Series]
    optional: bool = False


def _log_ret(s: pd.Series) -> pd.Series:
    s = s[s > 0].sort_index()
    return np.log(s).diff()


def _splice(new: pd.Series, old: pd.Series) -> pd.Series:
    new = new.dropna().sort_index()
    old = old.dropna().sort_index()
    if new.empty:
        return old
    first = new.index[0]
    ref = old.loc[:first]
    if ref.empty:
        return new
    scale = new.iloc[0] / ref.iloc[-1]
    return pd.concat([old.loc[old.index < first] * scale, new])


def _eq(lv, w):
    return _log_ret(lv["^GSPC"])


def _tech(lv, w):
    ixic, spx = _log_ret(lv["^IXIC"]), _log_ret(lv["^GSPC"])
    both = pd.concat([ixic, spx], axis=1, keys=["y", "x"]).dropna()
    if both.empty:
        return pd.Series(dtype=float)

    def beta(df: pd.DataFrame) -> float:
        d = float(np.dot(df["x"], df["x"]))
        return float(np.dot(df["x"], df["y"]) / d) if d > 0 else 0.0

    out = both["y"] - beta(both) * both["x"]
    for a, b in w or ():
        sub = both.loc[pd.Timestamp(a) : pd.Timestamp(b)]
        if len(sub) > 1:
            out.loc[sub.index] = sub["y"] - beta(sub) * sub["x"]
    return out


def _usd(lv, w):
    return _log_ret(_splice(lv["DTWEXBGS"], lv["DX-Y.NYB"]))


def _rates(lv, w):
    s = lv["DGS10"].dropna().sort_index()
    return -RATES_DURATION * s.diff() / 100.0


def _credit(lv, w):
    return lv["BAA10Y"].dropna().sort_index().diff() / 100.0


def _vol(lv, w):
    return _log_ret(lv["^VIX"])


def _haven(lv, w):
    return _log_ret(lv["GC=F"])


def _commodity(lv, w):
    s = lv["DCOILWTICO"].dropna().sort_index()
    s = s.where(s > 0)  # WTI printed negative once; returns are undefined there
    return s.pct_change(fill_method=None)


FACTOR_REGISTRY: dict[str, FactorDef] = {
    "EQ": FactorDef("EQ", ("^GSPC",), "Proxy: S&P 500 price index log return.", _eq),
    "TECH": FactorDef(
        "TECH",
        ("^IXIC", "^GSPC"),
        "Proxy: Nasdaq Composite log return orthogonalised to S&P 500 within each window.",
        _tech,
    ),
    "USD": FactorDef(
        "USD",
        ("DTWEXBGS", "DX-Y.NYB"),
        "Proxy: FRED broad dollar index spliced to the ICE dollar index before 2006.",
        _usd,
    ),
    "RATES": FactorDef(
        "RATES", ("DGS10",), "Proxy: -8 * change in 10-year Treasury yield, as a return.", _rates
    ),
    "CREDIT": FactorDef(
        "CREDIT", ("BAA10Y",), "Proxy: change in Baa minus 10-year Treasury spread.", _credit
    ),
    "VOL": FactorDef("VOL", ("^VIX",), "Proxy: change in log VIX.", _vol),
    "HAVEN": FactorDef(
        "HAVEN",
        ("GC=F",),
        "Proxy: gold futures log return; free series has a gap of about five months from 2000-03.",
        _haven,
    ),
    "COMMODITY": FactorDef(
        "COMMODITY", ("DCOILWTICO",), "Proxy: WTI crude oil return.", _commodity, optional=True
    ),
}


def _blank_long_jumps(r: pd.Series) -> pd.Series:
    """Drop returns that span a calendar gap too long to be a holiday."""
    if len(r) < 2:
        return r
    span = pd.Series(r.index, index=r.index).diff().dt.days
    return r.where(~(span > MAX_GAP_DAYS))


def _nan_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Boolean array marking NaN runs strictly longer than ``min_len``."""
    out = np.zeros(len(mask), dtype=bool)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i > min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DatetimeIndex], list[str]]:
    df = df.copy()
    flagged: dict[str, pd.DatetimeIndex] = {}
    warns: list[str] = []
    for col in df.columns:
        long_run = _nan_runs(df[col].isna().to_numpy(), MAX_GAP_BARS)
        if long_run.any():
            others = [c for c in df.columns if c != col]
            known = ~df[col].isna().to_numpy()
            fill = np.full(long_run.sum(), float(df[col].mean()) if known.any() else 0.0)
            if others and known.sum() > len(others) + 5:
                x = df[others].fillna(0.0).to_numpy()
                design = np.column_stack([np.ones(len(df)), x])
                coef, *_ = np.linalg.lstsq(design[known], df[col].to_numpy()[known], rcond=None)
                fill = design[long_run] @ coef
            df.loc[long_run, col] = fill
            flagged[col] = df.index[long_run]
            warns.append(
                f"{col}: {int(long_run.sum())} bars imputed with the conditional mean given the "
                "other factors (source data gap); results for this factor are model-filled"
            )
        df[col] = df[col].fillna(0.0)  # short gaps: instrument did not trade, zero return
    return df, flagged, warns


def _resample(df: pd.DataFrame, flags: dict, clock: Clock):
    bpy = float(clock.bars_per_year)
    if bpy >= 200:
        return df, flags
    if 40 <= bpy <= 60:
        period = "W-FRI"
    elif 10 <= bpy <= 14:
        period = "M"
    else:
        raise ValueError(
            f"bars_per_year={bpy} is not supported by the daily free-data proxies "
            "(use daily, weekly or monthly bars)"
        )
    key = df.index.to_period(period)
    out = df.groupby(key).sum(min_count=1)
    out.index = out.index.to_timestamp(how="end").normalize()
    new_flags = {}
    for col, idx in flags.items():
        hit = pd.Series(True, index=idx).groupby(idx.to_period(period)).any()
        stamp = hit.index.to_timestamp(how="end").normalize()
        new_flags[col] = pd.DatetimeIndex(stamp)
    return out, new_flags


def build_factor_returns(
    source: FactorProxySource,
    start: str,
    end: str,
    clock: Clock,
    factors: Sequence[str] | None = None,
    windows: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build the factor return panel from proxy series.

    Parameters
    ----------
    factors:
        Names from ``FACTOR_REGISTRY``. Defaults to every non-optional factor.
    windows:
        Calibration windows within which ``TECH`` is orthogonalised to ``EQ``.
        Defaults to the whole span.

    Notes
    -----
    Long source gaps are imputed and reported in ``df.attrs["imputed"]``
    (factor -> DatetimeIndex) and ``df.attrs["warnings"]``.
    """
    names = list(factors) if factors is not None else [
        n for n, f in FACTOR_REGISTRY.items() if not f.optional
    ]
    for n in names:
        if n not in FACTOR_REGISTRY:
            raise KeyError(f"unknown factor {n!r}; available: {sorted(FACTOR_REGISTRY)}")
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    fetch_start = (t0 - pd.Timedelta(days=_PAD_DAYS)).strftime("%Y-%m-%d")
    levels: Levels = {}
    for n in names:
        for key in FACTOR_REGISTRY[n].inputs:
            if key not in levels:
                levels[key] = source.get(key, fetch_start, end)
    wins = list(windows) if windows else [(start, end)]
    rets = {n: _blank_long_jumps(FACTOR_REGISTRY[n].build(levels, wins)) for n in names}
    rets = {n: r.loc[t0:t1] for n, r in rets.items()}
    base = max(rets.values(), key=lambda r: int(r.notna().sum())).index
    df = pd.DataFrame({n: rets[n].reindex(base) for n in names}).sort_index()
    df, flags, warns = _impute(df)
    df, flags = _resample(df, flags, clock)
    df.attrs["imputed"] = flags
    df.attrs["warnings"] = warns
    df.attrs["note"] = "all factors are free-data proxies; see crisispaths.factors"
    return df

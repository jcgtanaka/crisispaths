"""Data sources for factor proxy series.

A source maps a raw series key (a Yahoo Finance ticker such as ``^GSPC`` or a
FRED series id such as ``DGS10``) to a level series. Everything here uses
free data only. No pickles are written; the on-disk cache is parquet.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

DEFAULT_FRED_KEYS = frozenset({"DGS10", "BAA10Y", "DTWEXBGS", "DCOILWTICO"})


@runtime_checkable
class FactorProxySource(Protocol):
    """Anything that can return a level series for a raw series key."""

    def get(self, factor: str, start: str, end: str) -> pd.Series:  # pragma: no cover
        ...


def _clip(s: pd.Series, start: str, end: str) -> pd.Series:
    s = s.sort_index()
    return s.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class FixtureSource:
    """In-memory source for tests and offline use."""

    def __init__(self, series: Mapping[str, pd.Series]):
        self._series = {k: v.sort_index().astype(float) for k, v in series.items()}

    @classmethod
    def from_csv(cls, path: str | Path) -> FixtureSource:
        """Load a wide CSV: first column is the date, remaining columns are series."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls({c: df[c].dropna() for c in df.columns})

    def get(self, factor: str, start: str, end: str) -> pd.Series:
        if factor not in self._series:
            raise KeyError(f"series {factor!r} not in fixture; have {sorted(self._series)}")
        return _clip(self._series[factor], start, end)


Fetcher = Callable[[str, str, str, str], pd.Series]


def _default_fetcher(key: str, start: str, end: str, kind: str) -> pd.Series:
    if kind == "fred":
        try:
            from pandas_datareader import data as pdr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pandas-datareader is required for FRED data: pip install 'crisispaths[data]'"
            ) from exc
        df = pdr.DataReader(key, "fred", start, end)
        return df.iloc[:, 0].astype(float).dropna()
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for market data: pip install 'crisispaths[data]'"
        ) from exc
    df = yf.download(key, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float).dropna()


class YFinanceFredSource:
    """Yahoo Finance and FRED source with a parquet cache.

    ``yfinance`` and ``pandas_datareader`` are imported lazily, only when a
    series is actually downloaded.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        fred_keys: set[str] | frozenset[str] | None = None,
        fetcher: Fetcher | None = None,
    ):
        if cache_dir is None:
            from platformdirs import user_cache_dir

            cache_dir = Path(user_cache_dir("crisispaths"))
        self.cache_dir = Path(cache_dir)
        self.fred_keys = frozenset(fred_keys) if fred_keys is not None else DEFAULT_FRED_KEYS
        self._fetcher = fetcher or _default_fetcher

    def _path(self, key: str, start: str, end: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return self.cache_dir / f"{safe}__{start}__{end}.parquet"

    def get(self, factor: str, start: str, end: str) -> pd.Series:
        path = self._path(factor, start, end)
        if path.exists():
            return pd.read_parquet(path)["value"].rename(factor)
        kind = "fred" if factor in self.fred_keys else "yfinance"
        s = self._fetcher(factor, start, end, kind)
        s = _clip(pd.Series(s, dtype=float), start, end).rename(factor)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        s.rename("value").to_frame().to_parquet(path)
        return s

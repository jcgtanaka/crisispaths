"""Calendar-time to bar-count conversion.

Everything in crisispaths that is expressed in calendar time ("3M", "2Y",
"6W") is converted to a number of bars through a :class:`Clock`, so the same
scenario runs on daily, weekly, hourly or 24/7 data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
_UNIT_DAYS = {"D": 1.0, "W": 7.0, "M": DAYS_PER_YEAR / 12.0, "Y": DAYS_PER_YEAR}
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([DWMY])$")
_CALENDARS = ("business", "24/7")


def parse_duration(text: str) -> float:
    """Return the length of a duration string such as ``'3M'`` in calendar days."""
    if not isinstance(text, str):
        raise ValueError(f"invalid duration {text!r}: expected a string like '3M', '2Y', '0D'")
    m = _DURATION_RE.match(text.strip())
    if m is None:
        raise ValueError(
            f"invalid duration {text!r}: expected <number><unit> with unit in D, W, M, Y "
            "(for example '3M', '2Y', '6W', '0D')"
        )
    return float(m.group(1)) * _UNIT_DAYS[m.group(2)]


def bars_per_session(index: pd.DatetimeIndex) -> float:
    """Median number of bars per calendar date of ``index`` (1 for daily or coarser data)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return 1.0
    counts = pd.Series(1, index=idx).groupby(idx.normalize()).sum()
    return float(counts.median())


def session_slots(index: pd.DatetimeIndex, min_share: float = 0.5) -> list[int]:
    """Times of day (seconds after midnight) present in at least ``min_share`` of the sessions."""
    idx = pd.DatetimeIndex(index)
    sec = (idx.hour * 3600 + idx.minute * 60 + idx.second).to_numpy()
    n_sessions = idx.normalize().nunique()
    vals, cnt = np.unique(sec, return_counts=True)
    return [int(v) for v, c in zip(vals, cnt, strict=True) if c >= min_share * n_sessions]


@dataclass(frozen=True)
class Clock:
    """Maps calendar durations to bars.

    Parameters
    ----------
    bars_per_year:
        Number of bars in a calendar year. If omitted, ``calendar`` must be a
        ``DatetimeIndex`` from which it is inferred.
    calendar:
        ``'business'``, ``'24/7'`` or a ``DatetimeIndex`` to infer from.
    bars_per_day:
        Bars per session date. Inferred from a ``DatetimeIndex`` (median bars per
        date); otherwise ``bars_per_year`` divided by 252 (business) or 365
        (24/7) when that ratio is at least 1.5, else 1.
    overnight_gap:
        Whether the first bar of a session follows a gap (closed market).
        Inferred from the index; default True for ``'business'``, False for
        ``'24/7'``.
    """

    bars_per_year: float | None = None
    calendar: object = "business"
    bars_per_day: float | None = None
    overnight_gap: bool | None = None

    def __post_init__(self) -> None:
        cal = self.calendar
        if isinstance(cal, pd.DatetimeIndex):
            inferred = Clock.infer(cal)
            bpy = self.bars_per_year if self.bars_per_year is not None else inferred.bars_per_year
            object.__setattr__(self, "bars_per_year", bpy)
            object.__setattr__(self, "calendar", inferred.calendar)
            if self.bars_per_day is None:
                object.__setattr__(self, "bars_per_day", inferred.bars_per_day)
            if self.overnight_gap is None:
                object.__setattr__(self, "overnight_gap", inferred.overnight_gap)
            return
        if cal not in _CALENDARS:
            raise ValueError(f"invalid calendar {cal!r}: expected one of {_CALENDARS} or an index")
        if self.bars_per_year is None or not self.bars_per_year > 0:
            raise ValueError("bars_per_year must be a positive number")
        if self.bars_per_day is None:
            ratio = float(self.bars_per_year) / (252.0 if cal == "business" else 365.0)
            object.__setattr__(self, "bars_per_day", float(round(ratio)) if ratio >= 1.5 else 1.0)
        elif not self.bars_per_day >= 1:
            raise ValueError("bars_per_day must be >= 1")
        if self.overnight_gap is None:
            object.__setattr__(self, "overnight_gap", cal == "business")

    @property
    def dt(self) -> float:
        """Length of one bar in years."""
        return 1.0 / float(self.bars_per_year)

    @property
    def is_intraday(self) -> bool:
        """True when there is more than one bar per session date."""
        return float(self.bars_per_day) > 1.0

    def daily(self) -> Clock:
        """The daily clock of the same calendar (this clock when it is not intraday)."""
        if not self.is_intraday:
            return self
        return Clock(float(self.bars_per_year) / float(self.bars_per_day), self.calendar,
                     bars_per_day=1.0, overnight_gap=self.overnight_gap)

    def sessions(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Session date of every bar of ``index`` (the calendar date, midnight-normalised)."""
        return pd.DatetimeIndex(index).normalize()

    def vol_scale(self, annual_vol: float) -> float:
        """Convert an annualised volatility to a per-bar volatility."""
        return annual_vol * math.sqrt(self.dt)

    def to_bars(self, duration: str) -> int:
        """Convert a calendar duration string to a whole number of bars."""
        days = parse_duration(duration)
        return int(round(days / DAYS_PER_YEAR * float(self.bars_per_year)))

    @classmethod
    def infer(cls, index: pd.DatetimeIndex) -> Clock:
        """Infer a clock from the timestamps of a DatetimeIndex."""
        idx = pd.DatetimeIndex(index).sort_values()
        if len(idx) < 2:
            raise ValueError("need at least two timestamps to infer a clock")
        median_days = float(pd.Series(idx).diff().dropna().median() / pd.Timedelta(days=1))
        if median_days >= 80:
            return cls(4, "business")
        if median_days >= 27:
            return cls(12, "business")
        if median_days >= 5:
            return cls(52, "business")
        has_weekend = bool((idx.dayofweek >= 5).any())
        days = 365 if has_weekend else 252
        per_day = bars_per_session(idx)
        bars = int(round(days * per_day))
        calendar = "24/7" if has_weekend else "business"
        if per_day <= 1.0:
            return cls(bars, calendar)
        sess = idx.normalize()
        first = pd.Series(idx, index=idx).groupby(sess).min()
        last = pd.Series(idx, index=idx).groupby(sess).max()
        within = pd.Series(idx).diff().dropna()
        step = within[within < pd.Timedelta(hours=12)].median()
        gaps = (first.iloc[1:].to_numpy() - last.iloc[:-1].to_numpy())
        gap = pd.Timedelta(np.median(gaps)) if len(gaps) else pd.Timedelta(0)
        has_gap = bool(gap > 2 * step) if pd.notna(step) else calendar == "business"
        return cls(bars, calendar, bars_per_day=float(round(per_day)), overnight_gap=has_gap)

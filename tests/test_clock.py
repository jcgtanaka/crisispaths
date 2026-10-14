import math

import pandas as pd
import pytest

from crisispaths.clock import Clock


def test_to_bars_business_daily():
    c = Clock(bars_per_year=252)
    assert c.to_bars("0D") == 0
    assert c.to_bars("2Y") == 504
    assert c.to_bars("3M") == 63
    assert c.to_bars("6W") == 29


def test_to_bars_weekly():
    c = Clock(bars_per_year=52)
    assert c.to_bars("1Y") == 52
    assert c.to_bars("6W") == 6


def test_to_bars_crypto_24_7():
    c = Clock(bars_per_year=365, calendar="24/7")
    assert c.to_bars("1Y") == 365
    assert c.to_bars("4W") == 28


def test_dt_and_vol_scale():
    c = Clock(bars_per_year=252)
    assert c.dt == pytest.approx(1 / 252)
    assert c.vol_scale(0.2) == pytest.approx(0.2 / math.sqrt(252))


@pytest.mark.parametrize("bad", ["3Q", "abc", "", "-2Y", "3"])
def test_bad_duration_raises(bad):
    with pytest.raises(ValueError, match="duration"):
        Clock(bars_per_year=252).to_bars(bad)


def test_bad_calendar_raises():
    with pytest.raises(ValueError, match="calendar"):
        Clock(bars_per_year=252, calendar="moon")


def test_infer_daily_business():
    idx = pd.bdate_range("2020-01-01", periods=300)
    c = Clock.infer(idx)
    assert c.bars_per_year == 252
    assert c.calendar == "business"


def test_infer_weekly():
    idx = pd.date_range("2020-01-03", periods=100, freq="W-FRI")
    assert Clock.infer(idx).bars_per_year == 52


def test_infer_crypto_daily():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    c = Clock.infer(idx)
    assert c.bars_per_year == 365
    assert c.calendar == "24/7"


def test_infer_hourly_24_7():
    idx = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    c = Clock.infer(idx)
    assert c.bars_per_year == 365 * 24


def test_infer_minute_24_7():
    idx = pd.date_range("2020-01-01", periods=60 * 24 * 5, freq="min")
    assert Clock.infer(idx).bars_per_year == 365 * 24 * 60


def test_infer_hourly_business():
    days = pd.bdate_range("2020-01-06", periods=20)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9 + h) for d in days for h in range(7)])
    c = Clock.infer(idx)
    assert c.calendar == "business"
    assert c.bars_per_year == 252 * 7


def test_calendar_can_be_index():
    idx = pd.bdate_range("2020-01-01", periods=300)
    assert Clock(calendar=idx).bars_per_year == 252


def test_infer_needs_two_points():
    with pytest.raises(ValueError):
        Clock.infer(pd.DatetimeIndex(["2020-01-01"]))

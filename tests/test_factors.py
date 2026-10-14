import numpy as np
import pandas as pd
import pytest

from crisispaths.clock import Clock
from crisispaths.data import FixtureSource, YFinanceFredSource
from crisispaths.factors import FACTOR_REGISTRY, build_factor_returns


def make_levels(n=900, seed=0, gold_gap=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    eq_r = rng.normal(0.0003, 0.01, n)
    tech_idio = rng.normal(0, 0.006, n)
    levels = {
        "^GSPC": 1000 * np.exp(np.cumsum(eq_r)),
        "^IXIC": 2000 * np.exp(np.cumsum(1.3 * eq_r + tech_idio)),
        "DX-Y.NYB": 90 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
        "DGS10": 4.0 + np.cumsum(rng.normal(0, 0.03, n)),
        "BAA10Y": 2.0 + np.cumsum(rng.normal(0, 0.02, n)),
        "^VIX": 20 * np.exp(np.cumsum(rng.normal(0, 0.04, n))),
        "GC=F": 500 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
    }
    out = {k: pd.Series(v, index=idx) for k, v in levels.items()}
    start_new = pd.Timestamp("2006-01-02")
    new = 100 * out["DX-Y.NYB"][idx >= start_new] / out["DX-Y.NYB"][idx >= start_new].iloc[0]
    out["DTWEXBGS"] = new * 1.2
    if gold_gap is not None:
        s = out["GC=F"]
        out["GC=F"] = s[(s.index < gold_gap[0]) | (s.index > gold_gap[1])]
    return out, eq_r, tech_idio


def test_registry_contents():
    assert {"EQ", "TECH", "USD", "RATES", "CREDIT", "VOL", "HAVEN", "COMMODITY"} <= set(
        FACTOR_REGISTRY
    )
    assert FACTOR_REGISTRY["EQ"].inputs == ("^GSPC",)
    assert FACTOR_REGISTRY["COMMODITY"].optional
    assert not FACTOR_REGISTRY["EQ"].optional
    for f in FACTOR_REGISTRY.values():
        assert "proxy" in f.note.lower()
    # the HY OAS series must never be used
    assert all("BAMLH0A0HYM2" not in i for f in FACTOR_REGISTRY.values() for i in f.inputs)


def test_build_default_columns_and_values():
    lv, eq_r, _ = make_levels()
    src = FixtureSource(lv)
    df = build_factor_returns(src, "2006-03-01", "2008-06-30", Clock(252))
    assert list(df.columns) == ["EQ", "TECH", "USD", "RATES", "CREDIT", "VOL", "HAVEN"]
    gs = lv["^GSPC"]
    exp = np.log(gs).diff().loc[df.index]
    pd.testing.assert_series_equal(df["EQ"], exp, check_names=False, check_freq=False)
    dgs = lv["DGS10"]
    pd.testing.assert_series_equal(
        df["RATES"], -8.0 * dgs.diff().loc[df.index] / 100.0, check_names=False, check_freq=False
    )
    pd.testing.assert_series_equal(
        df["CREDIT"], lv["BAA10Y"].diff().loc[df.index] / 100.0, check_names=False, check_freq=False
    )
    assert df.index.min() >= pd.Timestamp("2006-03-01")
    assert df.index.max() <= pd.Timestamp("2008-06-30")


def test_tech_orthogonal_to_eq_within_window():
    lv, _, _ = make_levels()
    df = build_factor_returns(FixtureSource(lv), "2006-03-01", "2008-06-30", Clock(252))
    assert abs(np.dot(df["TECH"], df["EQ"])) < 1e-12
    w = [("2006-03-01", "2006-12-29"), ("2007-01-02", "2008-06-30")]
    df2 = build_factor_returns(
        FixtureSource(lv), "2006-03-01", "2008-06-30", Clock(252), windows=w
    )
    for a, b in w:
        sub = df2.loc[a:b]
        assert abs(np.dot(sub["TECH"], sub["EQ"])) < 1e-12


def test_usd_splice_is_continuous():
    lv, _, _ = make_levels()
    df = build_factor_returns(FixtureSource(lv), "2005-02-01", "2007-06-30", Clock(252))
    # the spliced level has no jump at the 2006 handover, so its returns match DX-Y.NYB
    dx = np.log(lv["DX-Y.NYB"]).diff()
    day = pd.Timestamp("2006-01-02")
    assert df.loc[day, "USD"] == pytest.approx(dx.loc[day], abs=1e-9)
    assert df["USD"].abs().max() < 0.05


def test_haven_gap_is_imputed_and_flagged():
    gap = (pd.Timestamp("2006-04-03"), pd.Timestamp("2006-09-29"))
    lv, _, _ = make_levels(gold_gap=gap)
    df = build_factor_returns(FixtureSource(lv), "2006-01-10", "2008-06-30", Clock(252))
    assert df["HAVEN"].notna().all()
    imputed = df.attrs["imputed"]["HAVEN"]
    assert len(imputed) > 100
    assert imputed.min() >= gap[0] and imputed.max() <= gap[1] + pd.Timedelta(days=7)
    assert any("HAVEN" in w for w in df.attrs["warnings"])


def test_optional_commodity_and_weekly_clock():
    lv, _, _ = make_levels()
    lv["DCOILWTICO"] = pd.Series(
        60 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 900))),
        index=lv["^GSPC"].index,
    )
    src = FixtureSource(lv)
    df = build_factor_returns(
        src, "2006-03-01", "2008-06-30", Clock(52), factors=["EQ", "COMMODITY"]
    )
    assert list(df.columns) == ["EQ", "COMMODITY"]
    assert len(df) < 140
    daily = build_factor_returns(
        src, "2006-03-01", "2008-06-30", Clock(252), factors=["EQ"]
    )
    assert df["EQ"].sum() == pytest.approx(daily["EQ"].sum(), abs=0.05)


def test_unknown_factor_and_missing_series():
    lv, _, _ = make_levels()
    with pytest.raises(KeyError):
        build_factor_returns(FixtureSource(lv), "2006-03-01", "2007-01-01", Clock(252),
                             factors=["NOPE"])
    del lv["^VIX"]
    with pytest.raises(KeyError):
        build_factor_returns(FixtureSource(lv), "2006-03-01", "2007-01-01", Clock(252))


def test_fixture_source_csv(tmp_path):
    lv, _, _ = make_levels()
    pd.DataFrame({"^GSPC": lv["^GSPC"]}).to_csv(tmp_path / "f.csv", index_label="date")
    src = FixtureSource.from_csv(tmp_path / "f.csv")
    s = src.get("^GSPC", "2005-01-10", "2005-01-20")
    assert s.index.min() >= pd.Timestamp("2005-01-10")
    assert s.index.max() <= pd.Timestamp("2005-01-20")


def test_yfinance_source_cache_and_dispatch(tmp_path):
    calls = []

    def fetcher(key, start, end, kind):
        calls.append((key, kind))
        idx = pd.bdate_range(start, end)
        return pd.Series(np.arange(len(idx), dtype=float) + 1.0, index=idx)

    src = YFinanceFredSource(cache_dir=tmp_path, fetcher=fetcher)
    a = src.get("^GSPC", "2020-01-01", "2020-02-01")
    b = src.get("^GSPC", "2020-01-01", "2020-02-01")
    pd.testing.assert_series_equal(a, b, check_freq=False)
    assert a.name == "^GSPC"
    assert calls == [("^GSPC", "yfinance")]
    src.get("DGS10", "2020-01-01", "2020-02-01")
    assert calls[-1] == ("DGS10", "fred")


@pytest.mark.network
def test_network_smoke(tmp_path):  # pragma: no cover
    src = YFinanceFredSource(cache_dir=tmp_path)
    s = src.get("^GSPC", "2020-01-01", "2020-02-01")
    assert len(s) > 10

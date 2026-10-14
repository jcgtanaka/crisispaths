"""Intraday: clock inference, profile estimation, disaggregation, wiring, regression."""

import dataclasses
import hashlib
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import crisispaths as cp
from conftest import CALM_WINDOWS
from crisispaths import Clock
from crisispaths.disaggregate import (
    IntradayProfile,
    aggregate_log_returns,
    aggregate_to_daily,
    disaggregate,
    estimate_intraday_profile,
)
from helpers_strategies import sma_trend
from intraday_fixtures import bridge_bars, intraday_prices, session_index, u_shape

SLOTS = 13
ASSETS5 = ["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]
INTRA_CLOCK = Clock(252 * SLOTS, "business")
MEMORY_BOUND_BYTES = 16 * 1024 * 1024  # measured peak 10.9 MiB; output arrays alone are 5.4 MiB


# --- clock -------------------------------------------------------------------------------------


def test_daily_clock_behaviour_unchanged():
    c = Clock(calendar=pd.bdate_range("2020-01-01", periods=300))
    assert (c.bars_per_year, c.bars_per_day, c.is_intraday) == (252, 1.0, False)
    assert c == Clock(252, "business")
    assert Clock.infer(pd.date_range("2020-01-31", periods=40, freq="ME")).bars_per_year == 12
    assert not Clock(52, "business").is_intraday


def test_intraday_inference_and_sessions():
    idx = session_index(pd.bdate_range("2021-03-01", periods=30), SLOTS)
    c = Clock(calendar=idx)
    assert c.bars_per_day == SLOTS and c.is_intraday and c.overnight_gap
    assert c.bars_per_year == 252 * SLOTS and c.calendar == "business"
    sess = c.sessions(idx)
    assert len(sess) == len(idx) and sess.nunique() == 30
    assert (sess == idx.normalize()).all()
    assert c.daily() == Clock(252, "business")


def test_247_hourly_has_no_overnight_gap():
    idx = pd.date_range("2021-01-01", periods=24 * 20, freq="h")
    c = Clock(calendar=idx)
    assert c.calendar == "24/7" and c.bars_per_day == 24 and not c.overnight_gap
    assert c.bars_per_year == 365 * 24


def test_explicit_intraday_clock_derives_bars_per_day():
    c = Clock(252 * 78, "business")
    assert c.bars_per_day == 78 and c.is_intraday
    assert c.daily() == Clock(252, "business")


# --- disaggregation ----------------------------------------------------------------------------


@pytest.mark.parametrize("shares", [
    np.full(13, 1 / 13), u_shape(13), u_shape(13, depth=10.0, gap_extra=0.5),
    np.r_[0.9, np.full(5, 0.02)],
])
def test_bars_sum_exactly_to_the_daily_return(shares):
    rng = np.random.default_rng(1)
    d = rng.normal(0, 0.02, size=(80, 3))
    prof = IntradayProfile.flat(len(shares), ["a", "b", "c"], daily_vol=np.full(3, 0.01))
    prof = dataclasses.replace(prof, weights=np.tile(shares, (3, 1)), pooled=shares)
    bars = disaggregate(d, prof, np.random.default_rng(2))
    assert bars.shape == (80 * len(shares), 3)
    np.testing.assert_allclose(bars.reshape(80, len(shares), 3).sum(axis=1), d, atol=1e-12)


def test_bar_variances_follow_the_profile():
    shares = u_shape(13, depth=5.0)
    d = np.zeros((6000, 1))
    prof = dataclasses.replace(
        IntradayProfile.flat(13, ["a"], daily_vol=np.array([0.01])),
        weights=shares[None, :], pooled=shares)
    bars = disaggregate(d, prof, np.random.default_rng(3), df=8.0).reshape(6000, 13)
    v = (bars**2).mean(axis=0)
    emp = v / v.sum()
    assert np.abs(emp - shares).max() < 0.012
    kurt = stats.kurtosis(bars[:, 6] / bars[:, 6].std())
    assert kurt > 0.5  # fat tails (t innovations)


def test_cross_asset_correlation_from_the_daily_residual_corr():
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    prof = IntradayProfile.flat(13, ["a", "b"], daily_vol=np.array([0.01, 0.01]))
    bars = disaggregate(np.zeros((3000, 2)), prof, np.random.default_rng(4), corr=corr, df=30.0)
    assert abs(np.corrcoef(bars.T)[0, 1] - 0.7) < 0.05


def test_coarser_bars_sum_daily_log_returns():
    d = np.random.default_rng(5).normal(0, 0.01, size=(104, 2))
    w = aggregate_log_returns(d, 5)
    assert w.shape == (20, 2)  # trailing partial block dropped
    np.testing.assert_allclose(w[3], d[15:20].sum(axis=0), atol=1e-15)
    with pytest.raises(ValueError):
        aggregate_log_returns(d, 0)


def test_flat_profile_is_uniform():
    p = IntradayProfile.flat(13, ["a", "b"])
    assert p.source == "flat" and p.weights.shape == (2, 13)
    np.testing.assert_allclose(p.weights, 1 / 13)


# --- profile estimation ------------------------------------------------------------------------


def _synthetic_panel(n_days, shares, assets=3, seed=0, drop=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    d = rng.normal(0, 0.01, size=(n_days, assets))
    bars = bridge_bars(d, shares, np.full(assets, 0.01), rng)
    px = pd.DataFrame(100 * np.exp(np.cumsum(bars, axis=0)),
                      index=session_index(dates, len(shares)),
                      columns=[f"X{j}" for j in range(assets)])
    if drop:
        keep = rng.random(len(px)) > drop
        px = px[keep]
    return px


@pytest.mark.parametrize("n_days, tol", [(60, 0.05), (250, 0.025)])
def test_profile_recovery(n_days, tol):
    shares = u_shape(SLOTS, gap_extra=0.6)
    px = _synthetic_panel(n_days, shares)
    prof = estimate_intraday_profile(px, Clock(calendar=px.index))
    assert prof.source == "estimated" and prof.slots == SLOTS
    assert prof.weights.sum(axis=1) == pytest.approx(1.0)
    assert np.abs(prof.pooled - shares).max() < tol
    assert prof.has_overnight_gap and 0.0 < prof.overnight_share.mean() < 0.6
    assert prof.n_sessions.min() >= n_days - 2


def test_overnight_share_recovered():
    shares = u_shape(SLOTS, depth=1.0, gap_extra=1.5)
    px = _synthetic_panel(400, shares, seed=3)
    prof = estimate_intraday_profile(px, Clock(calendar=px.index))
    k = np.arange(SLOTS)
    v_base = 1.0 + ((k - (SLOTS - 1) / 2) / ((SLOTS - 1) / 2)) ** 2
    true_share = 1.5 / (v_base.sum() + 1.5)  # the gap's share of the session variance
    assert abs(prof.overnight_share.mean() - true_share) < 0.04


def test_missing_bars_do_not_break_estimation_or_aggregation():
    shares = u_shape(SLOTS)
    px = _synthetic_panel(250, shares, seed=4, drop=0.05)
    prof = estimate_intraday_profile(px, Clock(calendar=px.index))
    assert np.abs(prof.pooled - shares).max() < 0.03
    daily = aggregate_to_daily(px, Clock(calendar=px.index))
    assert daily.index.is_monotonic_increasing and len(daily) == 250
    assert daily.notna().all().all()


def test_short_history_asset_shrinks_toward_pooled():
    shares = u_shape(SLOTS, depth=6.0)
    px = _synthetic_panel(300, shares, assets=3, seed=6)
    cut = px.index[px.index.normalize() < px.index.normalize()[-1] - pd.Timedelta(days=25)]
    px.loc[cut, "X2"] = np.nan  # X2 has about 25 sessions
    prof = estimate_intraday_profile(px, Clock(calendar=px.index), min_sessions=120)
    j = list(prof.assets).index("X2")
    assert prof.n_sessions[j] < 40 and 0.0 < prof.shrinkage[j] < 0.5
    raw_dist = np.abs(prof.raw_weights[j] - prof.pooled).sum()
    shr_dist = np.abs(prof.weights[j] - prof.pooled).sum()
    assert shr_dist < raw_dist
    assert prof.shrinkage[0] == 1.0


def test_df_estimated_from_fat_tails_and_default():
    shares = u_shape(SLOTS)
    px = _synthetic_panel(400, shares, seed=7)
    prof = estimate_intraday_profile(px, Clock(calendar=px.index))
    assert 4.5 <= prof.df <= 50.0


# --- wiring: calibration, generation -------------------------------------------------------


@pytest.fixture(scope="module")
def intra(market):
    daily = market["returns"][ASSETS5]
    px = intraday_prices(daily, SLOTS, seed=11)
    return px, market["prices"][ASSETS5], market["factors"]


@pytest.fixture(scope="module")
def cals(intra, spec_ai):
    px, dpx, factors = intra
    daily_clock = Clock(252, "business")
    ci = cp.calibrate(px, spec_ai, clock=INTRA_CLOCK, factors=factors,
                      calm_windows=CALM_WINDOWS)
    cd = cp.calibrate(dpx, spec_ai, clock=daily_clock, factors=factors,
                      calm_windows=CALM_WINDOWS)
    return ci, cd


def test_intraday_calibration_equals_daily_calibration(cals):
    ci, cd = cals
    assert ci.bars_per_year == 252 and ci.intraday is not None
    assert ci.intraday.source == "estimated" and ci.intraday.slots == SLOTS
    np.testing.assert_allclose(ci.betas_stress, cd.betas_stress, atol=1e-9)
    np.testing.assert_allclose(ci.resid_sigma_calm, cd.resid_sigma_calm, rtol=1e-9)
    assert cp.calibration_hash(ci) != cp.calibration_hash(cd)
    assert cd.intraday is None


def test_flat_fallback_warns_when_only_daily_prices(intra, spec_ai):
    _, dpx, factors = intra
    with pytest.warns(UserWarning, match="flat"):
        cal = cp.calibrate(dpx, spec_ai, clock=INTRA_CLOCK, factors=factors,
                           calm_windows=CALM_WINDOWS)
    assert cal.intraday.source == "flat" and cal.intraday.slots == SLOTS


def test_intraday_path_reaggregates_to_the_daily_path_exactly(cals, spec_ai):
    ci, cd = cals
    daily_clock = Clock(252, "business")
    for i in range(3):
        p = cp.generate_path(ci, spec_ai, i, 5, INTRA_CLOCK)
        q = cp.generate_path(ci, spec_ai, i, 5, daily_clock)
        n = len(q.returns)
        assert len(p.returns) == n * SLOTS and len(p.phase) == n * SLOTS
        agg = p.returns.to_numpy().reshape(n, SLOTS, -1).sum(axis=1)
        np.testing.assert_allclose(agg, q.returns.to_numpy(), atol=1e-12)
        fagg = p.factors.to_numpy().reshape(n, SLOTS, -1).sum(axis=1)
        np.testing.assert_allclose(fagg, q.factors.to_numpy(), atol=1e-12)
        assert p.meta["bars_per_day"] == SLOTS and p.meta["bars"] == n * SLOTS
        assert p.meta["onset_bars"] == q.meta["onset_bars"] * SLOTS


def test_intraday_determinism_prefix_and_batch(cals, spec_ai):
    ci, _ = cals
    a = cp.generate_path(ci, spec_ai, 2, 9, INTRA_CLOCK)
    b = cp.generate_path(ci, spec_ai, 2, 9, INTRA_CLOCK)
    assert a.returns.equals(b.returns) and a.factors.equals(b.factors)
    short = dataclasses.replace(spec_ai, horizon="1Y")
    c = cp.generate_path(ci, short, 2, 9, INTRA_CLOCK)
    n = len(c.returns)
    assert n == 252 * SLOTS
    np.testing.assert_array_equal(a.returns.to_numpy()[:n], c.returns.to_numpy())
    batch = cp.generate_paths(ci, spec_ai, 3, 9, INTRA_CLOCK)
    assert batch[2].returns.equals(a.returns)


def test_coarser_clock_aggregates_the_daily_path(cals, spec_ai):
    _, cd = cals
    daily_clock = Clock(252, "business")
    p = cp.generate_path(cd, spec_ai, 0, 3, daily_clock)
    for bpy, k in ((52, 5), (12, 21)):
        w = cp.generate_path(cd, spec_ai, 0, 3, Clock(bpy, "business"))
        want = aggregate_log_returns(p.returns.to_numpy(), k)
        np.testing.assert_allclose(w.returns.to_numpy(), want, atol=1e-12)
        assert len(w.returns) == len(p.returns) // k
    with pytest.raises(ValueError, match="multiple"):
        cp.generate_path(cd, spec_ai, 0, 3, Clock(100, "business"))


def test_monthly_aggregation_matches_a_monthly_calibrated_run(market, spec_ai, cals):
    """Horizon returns: daily paths summed to months vs a run calibrated on monthly data.

    The monthly calibration gets the daily stress loadings as overrides (a 31-observation
    monthly Dimson fit cannot estimate them), so the test isolates the generator. Month-end
    window boundaries still shift the location slightly (a few percent of the horizon move).
    """
    _, cd = cals
    px = market["prices"][ASSETS5]
    fac = market["factors"]
    ov = {a: {f: float(cd.betas_stress[i, j]) for j, f in enumerate(cd.factors)}
          for i, a in enumerate(cd.assets)}
    cm = cp.calibrate(px.resample("ME").last(), spec_ai, clock=Clock(12, "business"),
                      factors=fac.resample("ME").sum(), loading_overrides=ov)
    monthly = Clock(12, "business")
    n = 100
    a = np.array([cp.generate_path(cd, spec_ai, i, 1, monthly).returns["A045"].sum()
                  for i in range(n)])
    b = np.array([cp.generate_path(cm, spec_ai, i, 2, monthly).returns["A045"].sum()
                  for i in range(n)])
    assert stats.ks_2samp(a, b).pvalue > 0.01


def test_end_to_end_backtest_on_intraday_data(intra, market):
    px, _, factors = intra
    clock = INTRA_CLOCK

    def momentum(p):
        ma = p.rolling(40).mean()
        return (p > ma).astype(float).where(ma.notna()) / p.shape[1]

    for fam in ("ai_bubble", "us_bankruptcy"):
        spec = cp.compose(fam, complexity="minimal")
        res = cp.backtest(momentum, px, spec, n_paths=2, seed=1, clock=clock, factors=factors,
                          costs_bps=1.0, n_boot=20)
        t = res.table
        assert len(t) == 2 and np.isfinite(t["terminal_wealth"]).all()
        assert "time_to_recover_days" in t.columns
        h = int(t["horizon_bars"].iloc[0])
        assert h % SLOTS == 0 and h >= 252 * SLOTS
        rec = t[t["recovered"]]
        if len(rec):
            np.testing.assert_allclose(rec["time_to_recover_days"],
                                       rec["time_to_recover_bars"] / SLOTS)

    def leaky(p):
        return (p.shift(-1) > p).astype(float) / p.shape[1]

    spec = cp.compose("ai_bubble", complexity="minimal")
    with pytest.raises(cp.LookaheadError):
        cp.backtest(leaky, px, spec, n_paths=1, seed=1, clock=clock, factors=factors)


def test_harness_labels_simulated_bars_with_session_times(cals, intra, spec_ai):
    px, _, _ = intra
    ci, _ = cals
    p = cp.generate_path(ci, spec_ai, 0, 1, INTRA_CLOCK)
    from crisispaths.harness import run_strategy

    res = run_strategy(sma_trend, px[list(ci.assets)], p, clock=INTRA_CLOCK)
    idx = res.returns.index
    assert idx[0].normalize() > px.index[-1].normalize()
    assert idx[0].time() == px.index[0].time()
    assert (idx.normalize().value_counts() == SLOTS).all()
    assert idx.dayofweek.max() <= 4


# --- memory ------------------------------------------------------------------------------------


def test_one_5min_path_memory_stays_bounded(cal_ai, spec_ai):
    """One path of 10 assets x 504 sessions x 78 bars (about 39k bars) in bounded memory."""
    clock = Clock(252 * 78, "business")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # flat profile fallback
        tracemalloc.start()
        tracemalloc.reset_peak()
        p = cp.generate_path(cal_ai, spec_ai, 0, 0, clock)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    assert len(p.returns) == 504 * 78 and p.returns.shape[1] == 10
    assert peak < MEMORY_BOUND_BYTES, f"peak {peak / 2**20:.1f} MiB"


# --- regression: daily-only behaviour is bit-identical to the pre-intraday code ---------------

REGRESSION = {
    "ai_bubble": ("136c10c9c07cda14c443cb75b1ea82170fd835dcbc457b8fc2cedf6690cbefb9",
                  "2f7b7188b06ae5adb773feb2c3665a50edc90639f15a4e0c4c55597d4f63c5a2"),
    "us_bankruptcy": ("efd1e38a75f8a4ec81037dbd8e7016e6bd33cfaac434d81a4462e59b1ad0340a",
                      "bc8f00935f1276d3ae8c29dc43577746aec0d0a1cc242146379fb83b84116d76"),
}


@pytest.mark.parametrize("fam", ["ai_bubble", "us_bankruptcy"])
def test_daily_only_run_is_bit_identical_to_before(market, fam):
    clock = Clock(252, "business")
    px = market["prices"][ASSETS5]
    spec = cp.compose(fam)
    cal = cp.calibrate(px, spec, clock=clock, factors=market["factors"],
                       calm_windows=CALM_WINDOWS)
    h = hashlib.sha256()
    for i in range(3):
        p = cp.generate_path(cal, spec, i, 11, clock)
        h.update(np.ascontiguousarray(p.returns.to_numpy()).tobytes())
        h.update(np.ascontiguousarray(p.factors.to_numpy()).tobytes())
        h.update(p.phase.tobytes())
    res = cp.backtest(sma_trend, px, spec, n_paths=3, seed=11, clock=clock, calibration=cal,
                      costs_bps=2.0, n_boot=20)
    t = res.table.to_numpy(dtype=float)
    h.update(np.ascontiguousarray(np.nan_to_num(t, nan=-9.0)).tobytes())
    assert (cp.calibration_hash(cal), h.hexdigest()) == REGRESSION[fam]


# --- runner and report ---------------------------------------------------------------------


def test_runner_and_report_state_frequency_and_caveats(intra, tmp_path):
    from crisispaths.report import write_report
    from crisispaths.runner import run_mc

    px, _, factors = intra
    spec = cp.compose("ai_bubble", complexity="minimal")
    tab = run_mc(px, "helpers_strategies:sma_trend", spec, n_paths=2, seed=0, factors=factors,
                 out_dir=tmp_path, workers=1, costs_bps=1.0, check_lookahead=False)
    assert "time_to_recover_days" in tab.columns
    text = write_report(tab, tmp_path, n_boot=50, plots=False).read_text()
    assert "13 bars per session" in text and "3276 bars per year" in text
    assert "Brownian bridge" in text and "profile: estimated" in text
    assert "Epps effect" in text and "stop-losses" in text
    assert "no jumps inside the session" in text


def test_report_flags_flat_profile(intra, tmp_path):
    from crisispaths.report import write_report
    from crisispaths.runner import run_mc

    _, dpx, factors = intra
    spec = cp.compose("ai_bubble", complexity="minimal")
    with pytest.warns(UserWarning, match="flat"):
        tab = run_mc(dpx, "helpers_strategies:sma_trend", spec, n_paths=1, seed=0,
                     factors=factors, clock=INTRA_CLOCK, out_dir=tmp_path, workers=1,
                     costs_bps=1.0, check_lookahead=False)
    text = write_report(tab, tmp_path, n_boot=50, plots=False).read_text()
    assert "profile: flat" in text and "NO intraday history" in text

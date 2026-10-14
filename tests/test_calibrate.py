import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from conftest import CALM_WINDOWS, FACTOR_NAMES, make_market
from crisispaths import Calibration, calibrate
from crisispaths.data import FixtureSource
from crisispaths.spec import FactorShock, ScenarioSpec


def test_calibration_structure(cal_us, market):
    cal = cal_us
    assert isinstance(cal, Calibration)
    assert cal.assets == tuple(market["prices"].columns)
    assert cal.factors == tuple(FACTOR_NAMES)
    A, K = len(cal.assets), len(cal.factors)
    assert cal.betas_stress.shape == cal.betas_calm.shape == (A, K)
    assert cal.resid_corr_stress.shape == (A, A)
    assert cal.factor_cov_stress.shape == (K, K)
    assert cal.vol_cap.shape == cal.resid_sigma_calm.shape == cal.resid_ratio.shape == (A,)
    assert np.isfinite(cal.betas_stress).all()
    with pytest.raises(FrozenInstanceError):
        cal.assets = ()


def test_recovers_known_betas(cal_ai):
    j = cal_ai.factors.index("EQ")
    for a, truth in (("A045", 0.45), ("HIGH", 1.3), ("TECHY", 1.0)):
        assert cal_ai.betas_stress[cal_ai.assets.index(a), j] == pytest.approx(truth, abs=0.08)


def test_unaffected_flagged_and_zero(cal_us):
    assert "NOISE" in cal_us.unaffected
    i = cal_us.assets.index("NOISE")
    assert (cal_us.betas_stress[i] == 0).all()


def test_residual_corr_is_psd_unit_diagonal(cal_us):
    for c in (cal_us.resid_corr_stress, cal_us.resid_corr_calm):
        assert np.allclose(np.diag(c), 1.0)
        assert np.allclose(c, c.T)
        assert np.linalg.eigvalsh(c).min() > -1e-9


def test_vol_cap_is_max_rolling_std_in_windows(cal_us, market):
    r = np.log(market["prices"]).diff()["A045"]
    best = 0.0
    for a, b in cal_us.stress_windows:
        best = max(best, r.loc[a:b].rolling(40, min_periods=20).std().max())
    assert cal_us.vol_cap[cal_us.assets.index("A045")] == pytest.approx(best)


def test_stress_ratio_exceeds_one_where_noise_blows_up(cal_us):
    # synthetic idiosyncratic noise is doubled in stress windows
    assert cal_us.resid_ratio[cal_us.assets.index("A045")] > 1.5


def test_conditional_map_matches_ols(cal_us, spec_us):
    # default us spec anchors EQ only at default complexity? use whatever is anchored
    anch = list(cal_us.gamma.columns)
    assert set(anch) == set(spec_us.anchors)
    cond = list(cal_us.gamma.index)
    assert set(cond) == set(FACTOR_NAMES) - set(anch)
    assert cal_us.eta_cov.shape == (len(cond), len(cond))
    assert np.linalg.eigvalsh(cal_us.eta_cov.to_numpy()).min() > -1e-12


def test_analog_paths_hold_window_returns(cal_ai, market, spec_ai):
    w = spec_ai.analog_window
    assert w in cal_ai.analog_paths
    got = cal_ai.analog_paths[w]
    want = market["factors"].loc[w[0] : w[1]]
    assert np.allclose(got.to_numpy(), want.to_numpy())


def test_picklable_and_deterministic(cal_us, market, clock, spec_us):
    again = pickle.loads(pickle.dumps(cal_us))
    assert np.array_equal(again.betas_stress, cal_us.betas_stress)
    twin = calibrate(
        market["prices"], spec_us, clock=clock, factors=market["factors"],
        calm_windows=CALM_WINDOWS,
    )
    assert np.array_equal(twin.resid_corr_stress, cal_us.resid_corr_stress)
    assert np.array_equal(twin.vol_cap, cal_us.vol_cap)


def test_loading_overrides_applied(market, clock, spec_us):
    cal = calibrate(
        market["prices"], spec_us, clock=clock, factors=market["factors"],
        calm_windows=CALM_WINDOWS, loading_overrides={"A045": {"EQ": 2.0}},
    )
    assert cal.betas_stress[cal.assets.index("A045"), cal.factors.index("EQ")] == 2.0


def test_source_or_factors_required(market, clock, spec_us):
    with pytest.raises(ValueError, match="source"):
        calibrate(market["prices"], spec_us, clock=clock)


def test_anchor_missing_from_factors_is_error(market, clock, spec_us):
    spec = replace(spec_us, anchors={"NOPE": FactorShock("fixed", {"value": -0.1})})
    with pytest.raises(ValueError, match="NOPE"):
        calibrate(market["prices"], spec, clock=clock, factors=market["factors"])


def test_builds_factors_from_offline_source(clock):
    m = make_market(n_assets=3)
    f = m["factors"]
    spx = 1000.0 * np.exp(f["EQ"].cumsum())
    ixic = 1000.0 * np.exp((f["EQ"] * 1.2 + f["TECH"]).cumsum())
    src = FixtureSource({"^GSPC": spx, "^IXIC": ixic})
    spec = ScenarioSpec(
        name="t",
        calibration_windows=(("2008-09-01", "2009-03-09"),),
        anchors={
            "EQ": FactorShock("fixed", {"value": -0.3}),
            "TECH": FactorShock("fixed", {"value": -0.1}),
        },
    )
    cal = calibrate(
        m["prices"], spec, clock=clock, source=src, calm_windows=CALM_WINDOWS
    )
    assert set(cal.factors) == {"EQ", "TECH"}
    assert pd.notna(cal.betas_stress).all()


def test_nan_padded_assets_emit_no_runtime_warnings(market, clock, spec_ai):
    """Assets that start late (NaN prices before listing) must calibrate silently."""
    import warnings

    import numpy as np

    px = market["prices"][["A045", "HIGH", "BONDY", "GOLDY", "NOISE"]].copy()
    px.loc[px.index < "2004-11-18", "GOLDY"] = np.nan
    px.loc[px.index < "2014-09-17", "NOISE"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal = calibrate(px, spec_ai, clock=clock, factors=market["factors"])
    assert np.isfinite(cal.resid_sigma_calm).all() and np.isfinite(cal.resid_ratio).all()
    assert np.isfinite(cal.betas_stress).all()


# --- calm drift of the bootstrap pool ----------------------------------------------------------


@pytest.fixture(scope="module")
def drift_market(market):
    """The fixture market plus an asset with a strong calm drift (+10 bp per bar)."""
    px = market["prices"][["A045", "HIGH", "BONDY"]].copy()
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.006, len(px))
    px["DRIFT"] = 100.0 * np.exp(np.cumsum(r))
    return px


@pytest.fixture(scope="module")
def cal_pair(drift_market, market, clock, spec_ai):
    kw = dict(clock=clock, factors=market["factors"], calm_windows=CALM_WINDOWS)
    return (
        calibrate(drift_market, spec_ai, calm_drift="zero", **kw),
        calibrate(drift_market, spec_ai, calm_drift="historical", **kw),
    )


def test_default_is_zero_and_recorded(drift_market, market, clock, spec_ai, cal_pair):
    cal = calibrate(drift_market, spec_ai, clock=clock, factors=market["factors"],
                    calm_windows=CALM_WINDOWS)
    assert cal.calm_drift == "zero" and cal_pair[1].calm_drift == "historical"
    with pytest.raises(ValueError, match="calm_drift"):
        calibrate(drift_market, spec_ai, clock=clock, factors=market["factors"],
                  calm_drift="up")


def test_zero_pool_has_no_trend_but_same_spread(cal_pair):
    zero, hist = cal_pair
    assert np.abs(zero.calm_pool.mean(axis=0)).max() < 1e-12
    np.testing.assert_allclose(zero.calm_pool.std(axis=0), hist.calm_pool.std(axis=0),
                               rtol=1e-12)
    assert hist.calm_pool.mean(axis=0).max() > 5e-4  # the raw pool keeps the drift


def test_historical_pool_is_the_raw_calm_returns(drift_market, market, cal_pair):
    _, hist = cal_pair
    R = np.log(drift_market).diff().iloc[1:]
    F = market["factors"].loc[R.index]
    raw = pd.concat([R, F], axis=1)
    idx = raw.index
    mask = np.zeros(len(idx), bool)
    for a, b in CALM_WINDOWS:
        mask |= (idx >= a) & (idx <= b)
    stress = np.zeros(len(idx), bool)
    for a, b in hist.stress_windows:
        stress |= (idx >= a) & (idx <= b)
    want = raw.loc[mask & ~stress].fillna(0.0).to_numpy(float)
    np.testing.assert_allclose(hist.calm_pool, want, atol=1e-15)


def test_nan_padded_column_uses_its_own_valid_mean(market, clock, spec_ai):
    px = market["prices"][["A045", "BONDY", "GOLDY"]].copy()
    px.loc[px.index < "2010-01-01", "GOLDY"] = np.nan
    cal = calibrate(px, spec_ai, clock=clock, factors=market["factors"],
                    calm_windows=CALM_WINDOWS, calm_drift="zero")
    j = list(cal.assets).index("GOLDY")
    col = cal.calm_pool[:, j]
    valid = col != 0.0
    assert valid.sum() > 100 and abs(col[valid].mean()) < 1e-12


def test_calm_drift_changes_the_calibration_hash(cal_pair):
    import crisispaths as cp

    assert cp.calibration_hash(cal_pair[0]) != cp.calibration_hash(cal_pair[1])


def test_zero_drift_removes_trend_in_post_recovery_segment(cal_pair, clock):
    import crisispaths as cp

    spec = cp.load_scenario("ai_bubble_recovery")
    zero, hist = cal_pair

    def post_mean(cal):
        out = []
        for i in range(12):
            p = cp.generate_path(cal, spec, i, 0, clock)
            sel = p.phase == 3
            assert sel.sum() > 100
            out.append(p.returns["DRIFT"].to_numpy()[sel].mean())
        return float(np.median(out))

    assert abs(post_mean(zero)) < 3e-4
    assert post_mean(hist) > 6e-4


def test_scenarios_without_the_pool_are_bit_identical(cal_pair, clock):
    import crisispaths as cp

    zero, hist = cal_pair
    for name in ("ai_bubble_worst",):  # onset 0D, analog aftermath: the pool is never drawn
        spec = cp.load_scenario(name)
        assert spec.aftermath.mode != "v_recovery" and spec.onset == "0D"
        for i in range(3):
            a = cp.generate_path(zero, spec, i, 7, clock)
            b = cp.generate_path(hist, spec, i, 7, clock)
            assert a.returns.equals(b.returns) and a.factors.equals(b.factors)

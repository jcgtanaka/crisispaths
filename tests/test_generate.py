import time
import warnings
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import crisispaths as cp
from conftest import CALM_WINDOWS, make_market
from crisispaths import (
    AftermathSpec,
    Clock,
    FactorShock,
    calibrate,
    compose,
    describe_axes,
    generate_path,
    generate_paths,
)
from crisispaths._numerics import psd_corr
from crisispaths.generate import (
    dial_parameters,
    residual_scale,
    truncate_to_checkpoint,
)

SEED = 20240611


def gen(cal, spec, clock, i=0, **kw):
    return generate_path(cal, spec, i, SEED, clock, **kw)


def same(a, b):
    assert np.array_equal(a.returns.to_numpy(), b.returns.to_numpy())
    assert np.array_equal(a.factors.to_numpy(), b.factors.to_numpy())


# --- 1. determinism ------------------------------------------------------------------------


def test_same_seed_bit_identical(cal_us, spec_us, clock):
    same(gen(cal_us, spec_us, clock, 4), gen(cal_us, spec_us, clock, 4))
    other = generate_path(cal_us, spec_us, 4, SEED + 1, clock)
    mine = gen(cal_us, spec_us, clock, 4)
    assert not np.array_equal(other.returns.to_numpy(), mine.returns.to_numpy())


def test_path_i_alone_equals_path_i_in_batch(cal_us, spec_us, clock):
    batch = generate_paths(cal_us, spec_us, 6, SEED, clock)
    assert len(batch) == 6
    for i in (0, 3, 5):
        same(batch[i], gen(cal_us, spec_us, clock, i))
    assert not np.array_equal(batch[0].returns.to_numpy(), batch[1].returns.to_numpy())


# --- 2. prefix stability -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["analog", "prolonged_bear", "v_recovery", "stagnation"])
@pytest.mark.parametrize("analog_mode", ["exact", "block_bootstrap"])
def test_prefix_stability(cal_ai, spec_ai, clock, mode, analog_mode):
    base = replace(
        spec_ai,
        aftermath=replace(spec_ai.aftermath, mode=mode),
        analog_mode=analog_mode,
        onset=("0D", "1M"),
    )
    long = gen(cal_ai, replace(base, horizon="2Y"), clock, 3)
    short = gen(cal_ai, replace(base, horizon="1Y"), clock, 3)
    n = len(short.returns)
    assert n == clock.to_bars("1Y")
    assert np.array_equal(long.returns.to_numpy()[:n], short.returns.to_numpy())
    assert np.array_equal(long.factors.to_numpy()[:n], short.factors.to_numpy())
    cut = truncate_to_checkpoint(long, n)
    same(cut, short)


def test_truncate_bounds(cal_us, spec_us, clock):
    p = gen(cal_us, spec_us, clock)
    with pytest.raises(ValueError):
        truncate_to_checkpoint(p, 0)
    with pytest.raises(ValueError):
        truncate_to_checkpoint(p, len(p.returns) + 1)


# --- 4. anchor exactness ----------------------------------------------------------------


SHOCKS = {
    "fixed": FactorShock("fixed", {"value": -0.35}),
    "uniform": FactorShock("uniform", {"low": -0.5, "high": -0.2}),
    "lognormal_loss": FactorShock("lognormal_loss", {"median": 0.4, "sigma": 0.276, "cap": 0.95}),
}


@pytest.mark.parametrize("kind", list(SHOCKS))
def test_anchor_exactness(cal_us, spec_us, clock, kind):
    spec = replace(spec_us, anchors={"EQ": SHOCKS[kind]}, onset=("0D", "1M"))
    for i in range(8):
        p = gen(cal_us, spec, clock, i)
        pre, L = p.meta["onset_bars"], p.meta["crash_bars"]
        info = p.meta["anchors"]["EQ"]
        moved = p.factors["EQ"].iloc[pre : pre + L].sum()
        assert abs(moved - info["target"]) < 1e-12
        if kind == "fixed":
            assert info["target"] == pytest.approx(np.log1p(-0.35), abs=1e-15)
        if kind == "uniform":
            assert np.log1p(-0.5) <= info["target"] <= np.log1p(-0.2)
        if kind == "lognormal_loss":
            assert np.log1p(-0.95) <= info["target"] < 0


def test_loss_and_spread_scale(cal_us, spec_us, clock):
    base = replace(spec_us, anchors={"EQ": SHOCKS["fixed"]})
    scaled = replace(
        spec_us,
        anchors={"EQ": FactorShock("fixed", {"value": -0.35, "loss_scale": 1.25})},
    )
    a = gen(cal_us, base, clock).meta["anchors"]["EQ"]["target"]
    b = gen(cal_us, scaled, clock).meta["anchors"]["EQ"]["target"]
    assert b == pytest.approx(1.25 * a)


def test_units_rates_and_bp(cal_us, spec_us, clock):
    spec = replace(
        spec_us,
        anchors={
            "EQ": SHOCKS["fixed"],
            "RATES": FactorShock("fixed", {"value": 100, "unit": "bp"}),
            "CREDIT": FactorShock("fixed", {"value": 200, "unit": "bp"}),
            "VOL": FactorShock("fixed", {"value": 0.9, "unit": "log_change"}),
        },
    )
    p = gen(cal_us, spec, clock)
    t = {k: v["target"] for k, v in p.meta["anchors"].items()}
    assert t["RATES"] == pytest.approx(-8.0 * 0.01)
    assert t["CREDIT"] == pytest.approx(0.02)
    assert t["VOL"] == pytest.approx(0.9)


def test_haven_liquidation_and_post_drift(market, clock, cal_us):
    spec = compose("us_bankruptcy", complexity="full", usd="debasement")
    L = clock.to_bars("4W")
    for i in range(6):
        p = gen(cal_us, spec, clock, i)
        info = p.meta["anchors"]["HAVEN"]
        assert info["bars"] == L
        pre = p.meta["onset_bars"]
        assert abs(p.factors["HAVEN"].iloc[pre : pre + L].sum() - info["target"]) < 1e-12
        assert np.log1p(-0.25) <= info["target"] <= np.log1p(-0.10)
        assert 0.15 <= info["post_drift_annual"] <= 0.30


# --- 6. analog exact replay -----------------------------------------------------------------


def test_analog_exact_replay(cal_ai, spec_ai, clock, market):
    assert spec_ai.analog_mode == "exact" and spec_ai.aftermath.mode == "analog"
    w = spec_ai.analog_window
    win = market["factors"].loc[w[0] : w[1]]
    p = gen(cal_ai, spec_ai, clock)
    n = len(p.factors)
    for f in ("EQ", "TECH"):
        got = p.factors[f].cumsum().to_numpy()
        want = win[f].iloc[:n].cumsum().to_numpy()
        assert np.max(np.abs(got - want)) < 1e-9


def test_analog_keeps_realised_beta(cal_ai, spec_ai, clock):
    xs, ys = [], []
    for i in range(20):
        p = gen(cal_ai, spec_ai, clock, i)
        L = p.meta["crash_bars"]
        xs.append(p.factors["EQ"].iloc[:L].to_numpy())
        ys.append(p.returns["A045"].iloc[:L].to_numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    beta = np.cov(x, y)[0, 1] / x.var(ddof=1)
    assert beta == pytest.approx(0.45, abs=0.07)


# --- 8. PSD ----------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(dial=st.floats(min_value=0.0, max_value=1.25))
def test_dialed_correlation_is_psd(cal_us, dial):
    c = dial_parameters(cal_us, dial)["corr"]
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > -1e-9
    assert np.abs(c).max() <= 1.0 + 1e-9


@settings(max_examples=40, deadline=None)
@given(
    st.integers(2, 12).flatmap(
        lambda n: st.lists(
            st.floats(-1.5, 1.5), min_size=n * n, max_size=n * n
        ).map(lambda v: np.array(v).reshape(n, n))
    )
)
def test_psd_corr_repairs_anything(m):
    c = psd_corr(m + m.T)
    assert np.allclose(np.diag(c), 1.0)
    assert np.linalg.eigvalsh(c).min() > -1e-9


def test_dial_out_of_range_rejected(cal_us, spec_us, clock):
    for bad in (-0.1, 1.26):
        with pytest.raises(ValueError, match="dial"):
            gen(cal_us, spec_us, clock, dial=bad)


# --- model behaviour --------------------------------------------------------------------------


def test_unaffected_asset_has_no_factor_term(cal_us, spec_us, clock):
    i = cal_us.assets.index("NOISE")
    assert "NOISE" in cal_us.unaffected
    row = (np.arange(len(cal_us.assets)) == i)[:, None]
    tampered = replace(
        cal_us,
        betas_stress=np.where(row, 0.9, cal_us.betas_stress),
        betas_calm=np.where(row, 0.9, cal_us.betas_calm),
    )
    a, b = gen(cal_us, spec_us, clock, 2), gen(tampered, spec_us, clock, 2)
    assert np.array_equal(a.returns["NOISE"].to_numpy(), b.returns["NOISE"].to_numpy())
    assert a.meta["unaffected"] == ["NOISE"]


def test_residual_scale_respects_cap():
    base = np.array([0.02, 0.01, 0.03])
    qvar = np.array([1e-6, 4e-6, 0.0])
    cap = np.array([0.02, 0.015, np.inf])
    vmt = np.array([1.0, 3.0, 4.0])
    s = residual_scale(base, vmt, qvar, cap)
    total = np.sqrt(s**2 + (vmt**2)[:, None] * qvar)
    assert (total[:, :2] <= cap[:2] * (1 + 1e-12)).all()
    assert np.allclose(s[:, 2], base[2] * vmt)  # no cap, no change


def test_vol_cap_binds_in_generation(cal_us, clock, spec_us):
    i = cal_us.assets.index("A045")
    cap = np.full(len(cal_us.assets), np.inf)
    cap[i] = 0.0155
    capped = replace(cal_us, vol_cap=cap)
    spec = replace(spec_us, factor_df=8.0, copula_df=(8.0, 8.0), anchors={"EQ": SHOCKS["fixed"]})
    def pooled_var(cal):
        v = []
        for k in range(30):
            p = gen(cal, spec, clock, k)
            r = p.returns["A045"].iloc[: p.meta["crash_bars"]].to_numpy()
            v.append(r.var())
        return float(np.mean(v))
    assert pooled_var(capped) <= 0.0155**2 * 1.10
    assert pooled_var(capped) < pooled_var(cal_us)


@pytest.mark.parametrize("mode", ["analog", "prolonged_bear", "v_recovery", "stagnation"])
@pytest.mark.parametrize("n_assets", [1, 50])
def test_every_aftermath_mode_finite_right_length(clock, spec_ai, mode, n_assets):
    m = make_market(n_assets=n_assets)
    cal = calibrate(
        m["prices"], spec_ai, clock=clock, factors=m["factors"], calm_windows=CALM_WINDOWS
    ) if n_assets > 1 else calibrate(
        m["prices"][["A045"]], spec_ai, clock=clock, factors=m["factors"],
        calm_windows=CALM_WINDOWS,
    )
    spec = replace(spec_ai, aftermath=AftermathSpec(mode=mode), onset=("0D", "1M"))
    p = gen(cal, spec, clock, 1)
    A = 1 if n_assets == 1 else n_assets
    assert p.returns.shape == (clock.to_bars(spec.horizon), A)
    assert np.isfinite(p.returns.to_numpy()).all()
    assert np.isfinite(p.factors.to_numpy()).all()
    assert len(p.phase) == len(p.returns)


def test_aftermath_drift_shapes(cal_us, spec_us, clock):
    n = clock.to_bars("2Y")
    crash = clock.to_bars("3M")
    def eq_after(mode, **kw):
        spec = replace(
            spec_us, anchors={"EQ": SHOCKS["fixed"]}, onset="0D",
            aftermath=AftermathSpec(mode=mode, **kw),
        )
        tot = np.zeros(n - crash)
        for i in range(60):
            tot += gen(cal_us, spec, clock, i).factors["EQ"].to_numpy()[crash:]
        return tot / 60
    bear = eq_after("prolonged_bear", bear_annual_drift=-0.30)
    flat = eq_after("stagnation")
    rec = eq_after("v_recovery", recovery_fraction=(0.6, 0.6), recovery_period="9M")
    assert bear.sum() < flat.sum() - 0.1  # -30% annual over 1.75y
    assert abs(flat.sum()) < 0.15
    want = 0.6 * -np.log1p(-0.35)
    assert rec.sum() == pytest.approx(want, abs=0.12)


def test_onset_range_recorded_and_pre_segment_is_calm(cal_us, spec_us, clock):
    spec = replace(spec_us, onset=("1M", "2M"))
    lo, hi = clock.to_bars("1M"), clock.to_bars("2M")
    seen = set()
    for i in range(10):
        p = gen(cal_us, spec, clock, i)
        k = p.meta["onset_bars"]
        seen.add(k)
        assert lo <= k <= hi
        assert (p.phase[:k] == 0).all() and (p.phase[k : k + p.meta["crash_bars"]] == 1).all()
    assert len(seen) > 1
    assert gen(cal_us, replace(spec_us, onset="0D"), clock).meta["onset_bars"] == 0


@pytest.mark.parametrize("bpy, cal_type", [(52, "business"), (365, "24/7")])
def test_other_clocks_give_correct_bar_counts(cal_us, spec_us, bpy, cal_type):
    ck = Clock(bpy, cal_type)
    spec = replace(spec_us, anchors={"EQ": SHOCKS["fixed"]})
    p = generate_path(cal_us, spec, 0, SEED, ck)
    if bpy < cal_us.bars_per_year:
        # M6: a coarser clock on a daily calibration sums blocks of k = round(252 / bpy) daily
        # log returns (floor of the daily bar counts), instead of rescaling the daily model.
        k = round(cal_us.bars_per_year / bpy)
        daily = Clock(cal_us.bars_per_year, cal_type)
        assert len(p.returns) == daily.to_bars("2Y") // k
        assert p.meta["crash_bars"] == daily.to_bars("3M") // k
    else:
        assert len(p.returns) == ck.to_bars("2Y")
        assert p.meta["crash_bars"] == ck.to_bars("3M")
    assert np.isfinite(p.returns.to_numpy()).all()


def _single_axis_specs(family):
    axes = describe_axes(family)["axes"]
    out = {("default",): compose(family)}
    for axis, a in axes.items():
        for opt in a["options"]:
            out[(axis, opt)] = compose(family, **{axis: opt})
    return out


@pytest.mark.parametrize("family", ["ai_bubble", "us_bankruptcy"])
def test_every_composed_scenario_generates(cal_ai, cal_us, clock, family):
    cal = cal_ai if family == "ai_bubble" else cal_us
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.ScenarioWarning)
        specs = _single_axis_specs(family)
    assert len(specs) > 10
    for key, spec in specs.items():
        p = gen(cal, spec, clock, 0)
        assert len(p.returns) == clock.to_bars(spec.horizon), key
        assert np.isfinite(p.returns.to_numpy()).all(), key
        assert np.isfinite(p.factors.to_numpy()).all(), key


def test_missing_analog_window_is_clear_error(cal_us, spec_ai, clock):
    with pytest.raises(ValueError, match="analog"):
        gen(cal_us, spec_ai, clock)


def test_throughput_200_paths(cal_us, spec_us, clock):
    n, t0 = 200, time.perf_counter()
    out = generate_paths(cal_us, spec_us, n, SEED, clock)
    dt = time.perf_counter() - t0
    shape = out[0].returns.shape
    print(f"\nTHROUGHPUT {n} paths x {shape[1]} assets x {shape[0]} bars: {dt:.2f}s")
    assert len(out) == n and out[0].returns.shape == (504, 10)
    assert dt < 20.0


# --- conditional factors carry the crisis-conditional mean ------------------------------------


def _crash_moves(cal, spec, clock, n, factors):
    out = {f: [] for f in factors}
    for i in range(n):
        p = gen(cal, spec, clock, i)
        pre, L = p.meta["onset_bars"], p.meta["crash_bars"]
        for f in factors:
            out[f].append(p.factors[f].iloc[pre : pre + L].sum())
    return {f: np.array(v) for f, v in out.items()}


def test_conditional_mean_equals_gamma_times_anchored_mean(cal_us, spec_us, clock):
    spec = replace(spec_us, anchors={"EQ": SHOCKS["lognormal_loss"]}, conditional_factors=())
    g = float(cal_us.conditional_map(["EQ"])[0][
        [n for n in cal_us.factors if n != "EQ"].index("RATES"), 0])
    assert g < -0.1  # fixture: RATES co-moves negatively with EQ in stress
    mv = _crash_moves(cal_us, spec, clock, 300, ["EQ", "RATES"])
    diff = mv["RATES"] - g * mv["EQ"]  # = eta crash sum, mean zero if the mean is carried
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    assert abs(diff.mean()) < 3 * se
    assert mv["RATES"].mean() > 0.05  # historical sign: Treasuries rally in the crash


def test_history_consistent_rates_follow_history(cal_us, clock):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.ScenarioWarning)
        spec = compose("us_bankruptcy", complexity="full", rates_credit="history_consistent")
    assert "RATES" not in spec.anchors
    mv = _crash_moves(cal_us, spec, clock, 300, ["EQ", "RATES"])
    # EQ crash is a loss; RATES is -D*dy, so a rally (yields down) is a positive return
    assert mv["EQ"].mean() < -0.3
    assert mv["RATES"].mean() > 0.05
    names = list(cal_us.factors)
    anch = [n for n in names if n in spec.anchors]
    gam, _, cond = cal_us.conditional_map(anch)
    gk = gam[cond.index("RATES")]
    want = float(gk @ np.array([_crash_moves(cal_us, spec, clock, 300, anch)[a].mean()
                                for a in anch]))
    assert mv["RATES"].mean() == pytest.approx(want, abs=0.03)


def test_shock_rates_anchor_unaffected_by_conditional_fix(cal_us, clock):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.ScenarioWarning)
        spec = compose("us_bankruptcy", complexity="full", rates_credit="shock")
    for i in range(5):
        p = gen(cal_us, spec, clock, i)
        pre, L = p.meta["onset_bars"], p.meta["crash_bars"]
        t = p.meta["anchors"]["RATES"]["target"]
        assert abs(p.factors["RATES"].iloc[pre : pre + L].sum() - t) < 1e-12


@pytest.mark.parametrize("mode", ["prolonged_bear", "v_recovery", "stagnation"])
def test_conditional_factors_deterministic_and_prefix_stable(cal_us, spec_us, clock, mode):
    base = replace(
        spec_us, anchors={"EQ": SHOCKS["fixed"]}, onset=("0D", "1M"),
        aftermath=AftermathSpec(mode=mode),
    )
    a = gen(cal_us, replace(base, horizon="2Y"), clock, 2)
    b = gen(cal_us, replace(base, horizon="2Y"), clock, 2)
    s = gen(cal_us, replace(base, horizon="1Y"), clock, 2)
    cond = ["RATES", "CREDIT", "USD"]
    assert np.array_equal(a.factors[cond].to_numpy(), b.factors[cond].to_numpy())
    n = len(s.factors)
    assert np.array_equal(a.factors[cond].to_numpy()[:n], s.factors[cond].to_numpy())


def test_conditional_follows_anchor_through_aftermath(cal_us, spec_us, clock):
    spec = replace(
        spec_us, anchors={"EQ": SHOCKS["fixed"]}, onset="0D",
        aftermath=AftermathSpec(mode="prolonged_bear", bear_annual_drift=-0.4),
    )
    g = float(cal_us.conditional_map(["EQ"])[0][
        [n for n in cal_us.factors if n != "EQ"].index("RATES"), 0])
    L = clock.to_bars("3M")
    eq, rt = [], []
    for i in range(100):
        p = gen(cal_us, spec, clock, i)
        eq.append(p.factors["EQ"].iloc[L : L + 250].sum())
        rt.append(p.factors["RATES"].iloc[L : L + 250].sum())
    assert np.mean(rt) == pytest.approx(g * np.mean(eq), abs=0.05)

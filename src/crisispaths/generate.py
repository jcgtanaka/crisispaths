"""Path generation: one reproducible crisis path per call.

Model, per path (all quantities per bar of the clock)::

    anchored factors   f_a = drift_a + sigma_a * v_t * t_df(corr)      (t, unit variance)
    conditional        f_c = Gamma * f_a + eta_t   (f_a: the full simulated anchored path)
    assets             r_i = sum_k beta_ik f_k + eps_i
    residuals          eps_i = s_i * v_t * z_i,   z ~ t-copula with t margins

``Gamma`` is the stress-window regression of the unanchored factors on the
anchored ones without intercept, so conditional factors carry the
crisis-conditional mean (crash move and aftermath included), not only noise.
``pct`` units are fractions (-0.20 is -20%); ``percent`` is points (-20).

Crash phase: the anchored factor's cumulative move over the crash (or its
liquidation) window equals the drawn target exactly; innovations are de-meaned
and the target drift is added. ``v_t`` is the volatility envelope: the
``vol_multiplier`` draw (or the realised multiple of an analog replay) during
the crash, then a half-life decay back to one.

Aftermath modes: ``analog`` (replay or block bootstrap of the analog window),
``prolonged_bear`` (constant drift), ``v_recovery`` (a fraction of the log loss
is recovered, then joint calm bootstrap), ``stagnation`` (zero drift, vol floor).

Reproducibility: path ``i`` of seed ``s`` uses ``SeedSequence([s, i])`` and
does not depend on any other path. Every per-bar draw is prefix stable, so the
first ``h`` bars of a horizon-``H`` path equal the horizon-``h`` path.
Generation always runs at the calibration frequency; an intraday clock is
reached by Brownian-bridge disaggregation of the daily path (see
:mod:`crisispaths.disaggregate`) and coarser clocks by summing daily log returns.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._numerics import (
    block_rows,
    chunked,
    clip_psd,
    matrix_root,
    mvt,
    psd_corr,
    rows_matmul,
    stable_exp,
)
from .calibrate import Calibration
from .clock import Clock
from .disaggregate import IntradayProfile, aggregate_log_returns, disaggregate, warn_flat
from .factors import RATES_DURATION
from .spec import ScenarioSpec

MAX_DIAL = 1.25
CORR_LIMIT = 0.95
BLOCK = 10
STAGNATION_VOL_FLOOR = 1.5
CAP_FLOOR_FRACTION = 0.1  # residual scale never drops below this share of the cap
MIN_COPULA_DF = 3.0
PRIMARY_FACTOR = "EQ"
_ADDITIVE = ("CREDIT", "VOL")  # factor return is an additive change, not a log return
_UNIT_DIV = {None: 1.0, "pct": 1.0, "percent": 100.0, "bp": 1e4}
_N_STREAMS = 10


@dataclass
class Path:
    """One generated path.

    ``returns`` holds asset log returns, ``factors`` the factor returns, both
    indexed by synthetic bar number. ``phase`` is 0 before onset, 1 in the
    crash window, 2 in the aftermath and 3 in the post-recovery calm.
    """

    returns: pd.DataFrame
    factors: pd.DataFrame
    phase: np.ndarray
    vol_path: np.ndarray
    meta: dict = field(default_factory=dict)


GeneratedPath = Path


# --- calibration dial ---------------------------------------------------------------------------


def dial_parameters(
    cal: Calibration, m: float, copula_df: tuple[float, float] = (10.0, 4.0)
) -> dict[str, Any]:
    """Interpolate calm (``m=0``) to stress (``m=1``); ``m`` in (1, 1.25] extrapolates.

    Extrapolated correlations are clipped to ``+-0.95`` and pulled to PSD; the
    residual-scale ratio and degrees of freedom get floors so they stay valid.
    """
    if not 0.0 <= m <= MAX_DIAL:
        raise ValueError(f"dial must be in [0, {MAX_DIAL}], got {m}")
    lerp = lambda c, s: (1.0 - m) * c + m * s  # noqa: E731
    corr = lerp(cal.resid_corr_calm, cal.resid_corr_stress)
    ratio = lerp(np.ones_like(cal.resid_ratio), cal.resid_ratio)
    df = lerp(copula_df[0], copula_df[1])
    if m > 1.0:
        corr = clip_psd(corr, CORR_LIMIT)
        ratio = np.maximum(ratio, 0.1 * cal.resid_ratio)
        df = max(df, MIN_COPULA_DF)
    corr = psd_corr(corr) if m <= 1.0 else corr
    return {
        "corr": corr,
        "ratio": ratio,
        "df": float(df),
        "betas": lerp(cal.betas_calm, cal.betas_stress),
    }


def residual_scale(
    base: np.ndarray, vmt: np.ndarray, qvar: np.ndarray, cap: np.ndarray
) -> np.ndarray:
    """Residual std per bar and asset: ``base * v_t``, bounded so total vol stays under the cap.

    ``qvar`` is the asset's factor variance at unit envelope; the factor part at
    time t is ``v_t**2 * qvar``. A floor keeps the residual alive when the
    factor part alone exceeds the cap.
    """
    free = base[None, :] * vmt[:, None]
    cap2 = cap[None, :] ** 2
    with np.errstate(invalid="ignore"):
        room = np.sqrt(
            np.maximum(
                cap2 - (vmt**2)[:, None] * qvar[None, :],
                (CAP_FLOOR_FRACTION * cap[None, :]) ** 2,
            )
        )
    return np.minimum(free, room)


# --- anchors --------------------------------------------------------------------------------


def _native(name: str, unit: str | None, x: float) -> float:
    """Cumulative factor return implied by shock value ``x`` of factor ``name``."""
    if unit == "log_change":
        return float(x)
    if unit not in _UNIT_DIV:
        raise ValueError(f"unknown unit {unit!r} for factor {name!r}")
    d = float(x) / _UNIT_DIV[unit]
    if name == "RATES":
        return -RATES_DURATION * d
    if name in _ADDITIVE:
        return d
    if d <= -1.0:
        raise ValueError(
            f"factor {name!r}: a move of {x} {unit or 'pct'} is a loss of 100% or more"
        )
    return float(np.log1p(d))


def _target(name: str, shock, u: float, z: float) -> float:
    p = shock.params
    unit = p.get("unit")
    loss_scale = float(p.get("loss_scale", 1.0))
    spread_scale = float(p.get("spread_scale", 1.0))
    kind = shock.cum_dist
    if kind == "fixed":
        return _native(name, unit, p["value"]) * loss_scale * spread_scale
    if kind == "uniform":
        x = p["low"] + u * (p["high"] - p["low"])
        return _native(name, unit, x) * loss_scale * spread_scale
    if kind == "lognormal_loss":
        depth = min(p["median"] * np.exp(p["sigma"] * z) * loss_scale, p["cap"])
        return _native(name, unit, -depth) * spread_scale
    raise ValueError(f"cum_dist {kind!r} has no parametric target")  # pragma: no cover


# --- generation -----------------------------------------------------------------------------


def _bars(clock: Clock, duration: str, minimum: int = 0) -> int:
    return max(clock.to_bars(duration), minimum)


def _generate_native(
    cal: Calibration,
    spec: ScenarioSpec,
    i: int,
    seed: int,
    clock: Clock,
    *,
    dial: float = 1.0,
) -> Path:
    """Path at the calibration frequency (see :func:`generate_path`)."""
    if not 0.0 <= dial <= MAX_DIAL:
        raise ValueError(f"dial must be in [0, {MAX_DIAL}], got {dial}")
    anchors = spec.anchors
    if not anchors:
        raise ValueError("the spec has no anchors; at least one factor must be anchored")
    unknown = [n for n in [*anchors, *spec.conditional_factors] if n not in cal.factors]
    if unknown:
        raise ValueError(
            f"factor(s) {unknown} are not in the calibration (has {list(cal.factors)}); "
            "calibrate with a factor panel that contains them"
        )
    names = list(cal.factors)
    A, K = len(cal.assets), len(names)
    a_names = [n for n in names if n in anchors]
    a_idx = [names.index(n) for n in a_names]
    c_idx = [j for j in range(K) if j not in set(a_idx)]
    na = len(a_idx)
    primary = PRIMARY_FACTOR if PRIMARY_FACTOR in a_names else a_names[0]
    aft = spec.aftermath
    dt = clock.dt

    H = _bars(clock, spec.horizon, 1)
    crash_len = _bars(clock, spec.crash_length, 1)
    if isinstance(spec.onset, str):
        lo = hi = _bars(clock, spec.onset)
    else:
        lo, hi = _bars(clock, spec.onset[0]), _bars(clock, spec.onset[1])

    ss = np.random.SeedSequence([int(seed), int(i)])
    (s_sc, s_calm, s_fz, s_fc, s_az, s_ac, s_ez, s_ec, s_bb, s_post) = ss.spawn(_N_STREAMS)
    rng = np.random.default_rng(s_sc)
    pre = int(rng.integers(lo, hi + 1))
    v_draw = float(rng.uniform(*spec.vol_multiplier))
    rec_frac = float(rng.uniform(*aft.recovery_fraction))
    draws = {n: (float(rng.uniform()), float(rng.standard_normal()), float(rng.uniform()))
             for n in sorted(anchors)}

    n_total = max(H, pre + crash_len)
    n_post = n_total - pre
    n_aft = n_post - crash_len

    # --- analog window --------------------------------------------------------------------
    analog_names = [n for n in a_names if anchors[n].cum_dist == "analog"]
    need_window = bool(analog_names) or aft.mode == "analog"
    W = None
    if need_window:
        if not analog_names:
            raise ValueError("aftermath mode 'analog' needs at least one analog-anchored factor")
        if spec.analog_window is None:
            raise ValueError("an analog anchor needs spec.analog_window")
        win = cal.analog_paths.get(tuple(spec.analog_window))
        if win is None:
            raise ValueError(
                f"analog window {tuple(spec.analog_window)} is not in the calibration "
                f"(has {list(cal.analog_paths)}); calibrate with a spec that uses it"
            )
        Wfull = win[analog_names].to_numpy(float)
        if spec.analog_mode == "exact":
            need = n_post if aft.mode == "analog" else crash_len
            if len(Wfull) < need:
                raise ValueError(
                    f"exact analog replay needs {need} bars but the window has {len(Wfull)}"
                )
            W = Wfull[:n_post]
        else:
            W = block_rows(s_bb, n_post, Wfull, BLOCK)
    a_col = {n: q for q, n in enumerate(a_names)}
    w_col = {n: q for q, n in enumerate(analog_names)}

    # --- volatility envelope ------------------------------------------------------------------
    sig_f = cal.factor_sigma_calm[a_idx]
    if analog_names:
        ref = primary if primary in w_col else analog_names[0]
        sig_ref = cal.factor_sigma_calm[names.index(ref)]
        v = max(float(np.std(W[:crash_len, w_col[ref]]) / sig_ref), 1.0)
    else:
        v = v_draw
    hl = _bars(clock, aft.vol_half_life, 1)
    j = np.arange(1, n_aft + 1, dtype=float)
    vm_aft = 1.0 + (v - 1.0) * stable_exp(j * np.log(0.5) / hl)
    if aft.mode == "stagnation":
        vm_aft = np.maximum(vm_aft, min(STAGNATION_VOL_FLOOR, v))
    vmt = np.concatenate([np.full(crash_len, v), vm_aft])

    # --- anchored factor innovations ----------------------------------------------------------
    fdf = spec.factor_df
    cov_aa = cal.factor_cov_stress[np.ix_(a_idx, a_idx)]
    sd_a = np.sqrt(np.diag(cov_aa))
    corr_a = psd_corr(cov_aa / np.outer(sd_a, sd_a)) if na > 1 else np.eye(1)
    zn = chunked(s_fz, n_post, lambda r, c: r.standard_normal((c, na)))
    zc = chunked(s_fc, n_post, lambda r, c: r.chisquare(fdf, c))
    inn = mvt(zn, zc, matrix_root(corr_a), fdf) * sig_f[None, :] * vmt[:, None]

    f_a = np.zeros((n_post, na))
    info: dict[str, dict] = {}
    liq = {}
    for name in a_names:
        q = a_col[name]
        shock = anchors[name]
        p = shock.params
        u, z, u_post = draws[name]
        L = crash_len
        if "liquidation_length" in p:
            L = max(1, min(_bars(clock, p["liquidation_length"], 1), crash_len))
        liq[name] = L
        entry: dict[str, Any] = {"cum_dist": shock.cum_dist, "bars": L}
        if shock.cum_dist == "analog":
            scale = float(p.get("loss_scale", 1.0)) * float(p.get("spread_scale", 1.0))
            crash = W[:crash_len, w_col[name]] * scale
            f_a[:crash_len, q] = crash
            entry["bars"] = crash_len
            entry["target"] = float(crash.sum())
        else:
            t = _target(name, shock, u, z)
            seg = inn[:L, q]
            f_a[:L, q] = seg - seg.mean() + t / L
            entry["target"] = t
        if "post_drift_low" in p or "post_drift_high" in p:
            if "post_drift_low" not in p or "post_drift_high" not in p:
                raise ValueError(f"anchor {name!r}: post_drift_low and post_drift_high go together")
            unit = p.get("post_drift_unit", "annual")
            if unit != "annual":
                raise ValueError(f"anchor {name!r}: post_drift_unit must be 'annual', got {unit!r}")
            entry["post_drift_annual"] = p["post_drift_low"] + u_post * (
                p["post_drift_high"] - p["post_drift_low"]
            )
        info[name] = entry

    # aftermath drift: the market (primary) factor follows the mode; others only continue
    # their crash drift scaled by aftermath_drift_scale
    pq = a_col[primary]
    L_p = f_a[:crash_len, pq].sum()
    R = _bars(clock, aft.recovery_period, 1)
    if aft.mode == "prolonged_bear":
        mode_drift = np.full(n_aft, np.log1p(aft.bear_annual_drift) * dt)
    elif aft.mode == "v_recovery":
        mode_drift = np.where(j <= R, rec_frac * -L_p / R, 0.0)
    else:
        mode_drift = np.zeros(n_aft)
    for name in a_names:
        q = a_col[name]
        shock, L, entry = anchors[name], liq[name], info[name]
        d = np.zeros(n_post)
        if "post_drift_annual" in entry:
            d[L:] += np.log1p(entry["post_drift_annual"]) * dt
        scale = shock.aftermath_drift_scale
        if name == primary:
            d[crash_len:] += mode_drift * (scale if scale else 1.0)
        elif scale:
            d[crash_len:] += scale * entry["target"] / entry["bars"]
        replay = shock.cum_dist == "analog"
        start = crash_len if replay else L
        f_a[start:, q] = inn[start:, q] + d[start:]
        if replay and aft.mode == "analog":
            f_a[crash_len:, q] = W[crash_len:n_post, w_col[name]]

    # --- conditional factors ------------------------------------------------------------------
    f_all = np.zeros((n_post, K))
    f_all[:, a_idx] = f_a
    vol_ref = float(np.sqrt(np.mean(cal.factor_vol_ratio[a_idx] ** 2)))
    vol_ref = max(vol_ref, 1.0)
    if c_idx:
        gamma, eta_cov, _ = cal.conditional_map(a_names)
        en = chunked(s_ez, n_post, lambda r, c: r.standard_normal((c, len(c_idx))))
        ec = chunked(s_ec, n_post, lambda r, c: r.chisquare(fdf, c))
        eta = mvt(en, ec, matrix_root(eta_cov), fdf) * (vmt / vol_ref)[:, None]
        f_all[:, c_idx] = rows_matmul(f_a, gamma.T) + eta

    # --- assets --------------------------------------------------------------------------------
    dp = dial_parameters(cal, dial, spec.copula_df)
    B = np.nan_to_num(dp["betas"].copy())
    unaff = [cal.assets.index(a) for a in cal.unaffected]
    B[unaff] = 0.0
    qvar = np.einsum("ik,kl,il->i", B, cal.factor_cov_calm, B)
    vr = 1.0 + dial * (vol_ref - 1.0)
    base = cal.resid_sigma_calm * dp["ratio"] / vr
    scale = residual_scale(base, vmt, qvar, cal.vol_cap)
    an = chunked(s_az, n_post, lambda r, c: r.standard_normal((c, A)))
    ac = chunked(s_ac, n_post, lambda r, c: r.chisquare(dp["df"], c))
    eps = mvt(an, ac, matrix_root(dp["corr"]), dp["df"]) * scale
    r_a = rows_matmul(f_all, B.T) + eps

    # --- v_recovery: joint calm after the recovery window ------------------------------
    phase_post = np.concatenate([np.ones(crash_len, int), np.full(n_aft, 2, int)])
    post0 = crash_len + R
    if aft.mode == "v_recovery" and post0 < n_post:
        rows = block_rows(s_post, n_post - post0, cal.calm_pool, BLOCK)
        r_a[post0:], f_all[post0:] = rows[:, :A], rows[:, A:]
        vmt = vmt.copy()
        vmt[post0:] = 1.0
        phase_post[post0:] = 3

    # --- pre-onset calm -----------------------------------------------------------------------
    if pre:
        rows = block_rows(s_calm, pre, cal.calm_pool, BLOCK)
        r_full = np.concatenate([rows[:, :A], r_a])
        f_full = np.concatenate([rows[:, A:], f_all])
    else:
        r_full, f_full = r_a, f_all
    phase = np.concatenate([np.zeros(pre, int), phase_post])[:H]
    vol_path = np.concatenate([np.ones(pre), vmt])[:H]
    index = pd.RangeIndex(1, H + 1, name="bar")
    meta = {
        "seed": int(seed),
        "path": int(i),
        "bars": H,
        "onset_bars": pre,
        "crash_bars": crash_len,
        "vol_multiplier": v,
        "aftermath": aft.mode,
        "recovery_fraction": rec_frac if aft.mode == "v_recovery" else None,
        "primary_factor": primary,
        "anchors": info,
        "conditional": [names[k] for k in c_idx],
        "unaffected": list(cal.unaffected),
        "dial": float(dial),
        "scenario": spec.key,
        "label": spec.label,
    }
    return Path(
        returns=pd.DataFrame(r_full[:H], index=index, columns=list(cal.assets)),
        factors=pd.DataFrame(f_full[:H], index=index, columns=names),
        phase=phase,
        vol_path=vol_path,
        meta=meta,
    )


_INTRADAY_STREAM = 0x1D7A
_FREQ_TOL = 0.1


def _plan(cal: Calibration, clock: Clock) -> tuple[str, int]:
    """``("native", 1)``, ``("intraday", bars_per_day)`` or ``("coarse", k)``."""
    cal_bpy = float(cal.bars_per_year)
    if clock.is_intraday:
        daily = float(clock.bars_per_year) / float(clock.bars_per_day)
        if abs(daily / cal_bpy - 1.0) > _FREQ_TOL:
            raise ValueError(
                f"the intraday clock has {daily:g} sessions per year but the calibration has "
                f"{cal_bpy:g} bars per year; calibrate on data of the same calendar"
            )
        return "intraday", int(round(float(clock.bars_per_day)))
    if float(clock.bars_per_year) < cal_bpy * (1.0 - _FREQ_TOL):
        k = int(round(cal_bpy / float(clock.bars_per_year)))
        if abs(k * float(clock.bars_per_year) / cal_bpy - 1.0) > _FREQ_TOL:
            raise ValueError(
                f"clock has {clock.bars_per_year:g} bars per year, not a whole multiple of "
                f"1/{k} of the calibration's {cal_bpy:g}; use a clock that is an integer "
                "fraction of the calibration frequency"
            )
        return "coarse", k
    return "native", 1


def _to_coarse(p: Path, k: int) -> Path:
    n = (len(p.returns) // k) * k
    if n == 0:
        raise ValueError("the horizon is shorter than one coarse bar")
    m = n // k
    idx = pd.RangeIndex(1, m + 1, name="bar")
    meta = dict(p.meta)
    meta.update(bars=m, onset_bars=p.meta["onset_bars"] // k, crash_bars=p.meta["crash_bars"] // k,
                aggregated_from_bars=k)
    return Path(
        returns=pd.DataFrame(aggregate_log_returns(p.returns.to_numpy(), k), index=idx,
                             columns=p.returns.columns),
        factors=pd.DataFrame(aggregate_log_returns(p.factors.to_numpy(), k), index=idx,
                             columns=p.factors.columns),
        phase=p.phase[:n].reshape(m, k)[:, -1].copy(),
        vol_path=p.vol_path[:n].reshape(m, k).mean(axis=1),
        meta=meta,
    )


def _to_intraday(cal: Calibration, p: Path, i: int, seed: int, slots: int, clock: Clock) -> Path:
    prof = cal.intraday
    if prof is None or prof.slots != slots:
        if prof is not None:
            raise ValueError(
                f"the calibration's intraday profile has {prof.slots} bars per session but the "
                f"clock has {slots}"
            )
        warn_flat(slots)
        prof = IntradayProfile.flat(slots, cal.assets, has_overnight_gap=bool(clock.overnight_gap))
    names = list(map(str, cal.assets))
    if list(prof.assets) != names:
        order = [list(prof.assets).index(a) for a in names]
        prof = dataclasses.replace(
            prof, assets=tuple(names), weights=prof.weights[order],
            raw_weights=prof.raw_weights[order],
            n_sessions=prof.n_sessions[order], shrinkage=prof.shrinkage[order],
            overnight_share=prof.overnight_share[order], daily_vol=prof.daily_vol[order],
        )
    n = len(p.returns)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(i), _INTRADAY_STREAM]))
    bars = disaggregate(p.returns.to_numpy(), prof, rng, corr=cal.resid_corr_calm)
    pooled = prof.weights.mean(axis=0)
    pooled = pooled / pooled.sum()
    fac = (p.factors.to_numpy()[:, None, :] * pooled[None, :, None]).reshape(n * slots, -1)
    idx = pd.RangeIndex(1, n * slots + 1, name="bar")
    meta = dict(p.meta)
    meta.update(
        bars=n * slots, onset_bars=p.meta["onset_bars"] * slots,
        crash_bars=p.meta["crash_bars"] * slots, bars_per_day=slots, daily_bars=n,
        intraday_profile=prof.source,
    )
    return Path(
        returns=pd.DataFrame(bars, index=idx, columns=p.returns.columns),
        factors=pd.DataFrame(fac, index=idx, columns=p.factors.columns),
        phase=np.repeat(p.phase, slots),
        vol_path=np.repeat(p.vol_path, slots),
        meta=meta,
    )


def generate_path(
    cal: Calibration,
    spec: ScenarioSpec,
    i: int,
    seed: int,
    clock: Clock,
    *,
    dial: float = 1.0,
) -> Path:
    """Generate path ``i`` of ``seed``. Pure function of its arguments.

    ``dial`` in ``[0, 1.25]`` interpolates asset dependence, loadings and
    residual scale from calm (0) to the calibrated stress (1).

    The path is always simulated at the calibration frequency. When ``clock``
    is intraday it is then disaggregated into bars (the bars of a session sum
    to the daily return, see :mod:`crisispaths.disaggregate`); when ``clock`` is
    an integer fraction of the calibration frequency (weekly, monthly from
    daily) daily log returns are summed. Bar counts in ``meta`` are in bars of
    ``clock``. A flat intraday profile (with a ``UserWarning``) is used when the
    calibration holds none.
    """
    mode, k = _plan(cal, clock)
    if mode == "native":
        return _generate_native(cal, spec, i, seed, clock, dial=dial)
    base = Clock(float(cal.bars_per_year), clock.calendar if isinstance(clock.calendar, str)
                 else "business")
    p = _generate_native(cal, spec, i, seed, base, dial=dial)
    if mode == "coarse":
        return _to_coarse(p, k)
    return _to_intraday(cal, p, i, seed, k, clock)


def generate_paths(
    cal: Calibration,
    spec: ScenarioSpec,
    n: int,
    seed: int,
    clock: Clock,
    *,
    dial: float = 1.0,
) -> list[Path]:
    """Paths ``0..n-1``; each equals ``generate_path(cal, spec, i, seed, clock)`` alone."""
    return [generate_path(cal, spec, i, seed, clock, dial=dial) for i in range(n)]


def truncate_to_checkpoint(p: Path, bars: int) -> Path:
    """First ``bars`` bars of a path; identical to generating with that horizon."""
    n = len(p.returns)
    if not 1 <= bars <= n:
        raise ValueError(f"bars must be within the generated horizon 1..{n}")
    meta = dict(p.meta)
    meta["bars"] = bars
    return Path(
        returns=p.returns.iloc[:bars].copy(),
        factors=p.factors.iloc[:bars].copy(),
        phase=p.phase[:bars].copy(),
        vol_path=p.vol_path[:bars].copy(),
        meta=meta,
    )

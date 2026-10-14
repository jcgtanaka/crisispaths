"""Calibration: everything the path generator needs, estimated once.

:func:`calibrate` turns a price panel, a scenario spec and a factor panel into
a :class:`Calibration` made only of plain numpy and pandas objects. Generating
paths from it needs no network and no further estimation.

What is estimated
-----------------
* factor loadings per regime (``loadings.estimate_loadings``);
* factor means and covariance in the stress windows, and factor volatility in
  calm periods (the unit in which crash volatility multiples are expressed);
* the conditional-factor map ``Gamma`` (OLS, no intercept, of the unanchored
  factors on the anchored ones in stress windows) and the covariance of its residual ``eta``;
* per-asset residual volatility in calm and stress, their ratio, and the
  residual correlation matrices shrunk (Ledoit-Wolf) to constant correlation;
* a per-asset volatility cap: the largest 40-bar rolling standard deviation the
  asset itself showed inside the calibration windows;
* a pool of calm rows for block bootstrap and the factor paths of the windows
  used for analog replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ._numerics import ledoit_wolf_constant_corr, psd_corr
from .clock import Clock, bars_per_session
from .data import FactorProxySource
from .disaggregate import (
    IntradayProfile,
    aggregate_to_daily,
    estimate_intraday_profile,
    warn_flat,
)
from .factors import FACTOR_REGISTRY, build_factor_returns
from .loadings import Loadings, estimate_loadings
from .spec import ScenarioSpec

CAP_WINDOW = 40
CAP_MIN_PERIODS = 20


def conditional_map(
    cov: np.ndarray, names: Sequence[str], anchored: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Regression of every non-anchored factor on the anchored ones, WITHOUT intercept.

    ``cov`` is the uncentred second-moment matrix of the factors in the stress
    windows, so the fitted map carries the crisis-conditional mean:
    ``f_c = Gamma @ f_a + eta`` (as ``U = gamma * F + noise``).

    Returns ``(Gamma, eta_cov, conditional_names)`` with ``Gamma`` of shape
    ``(n_conditional, n_anchored)`` (anchored order as given) and ``eta_cov``
    the covariance of the regression residual.
    """
    names = list(names)
    a_idx = [names.index(n) for n in anchored]
    c_names = [n for n in names if n not in set(anchored)]
    c_idx = [names.index(n) for n in c_names]
    if not c_idx:
        return np.zeros((0, len(a_idx))), np.zeros((0, 0)), []
    cov = np.asarray(cov, dtype=float)
    saa = cov[np.ix_(a_idx, a_idx)]
    sca = cov[np.ix_(c_idx, a_idx)]
    gamma = np.linalg.lstsq(saa, sca.T, rcond=None)[0].T
    eta = cov[np.ix_(c_idx, c_idx)] - gamma @ saa @ gamma.T
    eta = (eta + eta.T) / 2.0
    w, v = np.linalg.eigh(eta)
    eta = (v * np.clip(w, 0.0, None)) @ v.T
    return gamma, eta, c_names


@dataclass(frozen=True)
class Calibration:
    """Frozen, picklable calibration. Arrays are indexed ``assets`` x ``factors``.

    ``calm_pool`` holds the calm-window rows (assets then factors) that the
    generator block-bootstraps after a V recovery and before the onset.
    ``calm_drift`` says how they were prepared: ``"zero"`` (default) subtracts
    each column's mean over its valid calm bars, so the blocks keep volatility,
    correlation and tails but carry no trend; ``"historical"`` keeps the raw
    returns, including whatever drift the calm windows had (a bull run in an
    asset then reappears as a calm-period gain in every path).
    """

    assets: tuple[str, ...]
    factors: tuple[str, ...]
    stress_windows: tuple[tuple[str, str], ...]
    calm_windows: tuple[tuple[str, str], ...]
    betas_calm: np.ndarray
    betas_stress: np.ndarray
    unaffected: tuple[str, ...]
    resid_sigma_calm: np.ndarray
    resid_ratio: np.ndarray  # stress / calm residual std, per asset
    resid_corr_calm: np.ndarray
    resid_corr_stress: np.ndarray
    vol_cap: np.ndarray
    factor_mean_stress: np.ndarray
    factor_cov_stress: np.ndarray
    factor_m2_stress: np.ndarray  # uncentred second moments, stress windows
    factor_cov_calm: np.ndarray
    factor_sigma_calm: np.ndarray
    factor_vol_ratio: np.ndarray  # stress / calm factor std
    gamma: pd.DataFrame  # conditional x anchored, for the spec calibrated with
    eta_cov: pd.DataFrame
    calm_pool: np.ndarray  # rows: assets then factors, calm bars
    analog_paths: dict = field(default_factory=dict)  # (start, end) -> factor returns frame
    last_price: pd.Series | None = None
    bars_per_year: float = 252.0
    shrinkage: tuple[float, float] = (np.nan, np.nan)  # calm, stress
    loadings: Loadings | None = None
    warnings: tuple[str, ...] = ()
    source: str | None = None
    calm_drift: str = "zero"
    intraday: IntradayProfile | None = None

    def conditional_map(self, anchored: Sequence[str]):
        """``(Gamma, eta_cov, conditional_names)`` for an arbitrary anchored set."""
        return conditional_map(self.factor_m2_stress, self.factors, anchored)


def _window_mask(index: pd.DatetimeIndex, wins: Sequence[tuple[str, str]]) -> np.ndarray:
    m = np.zeros(len(index), dtype=bool)
    for a, b in wins:
        m |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return m


def _runs(index: pd.DatetimeIndex, mask: np.ndarray) -> list[tuple[str, str]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((index[i].strftime("%Y-%m-%d"), index[j].strftime("%Y-%m-%d")))
            i = j + 1
        else:
            i += 1
    return out


def _nanstd(x: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Column standard deviation ignoring NaN; NaN (no warning) for columns with <= ddof values."""
    ok = ~np.isnan(x)
    cnt = ok.sum(axis=0)
    mean = np.where(cnt > 0, np.where(ok, x, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    dev = np.where(ok, x - mean, 0.0)
    var = (dev * dev).sum(axis=0) / np.maximum(cnt - ddof, 1)
    return np.where(cnt > ddof, np.sqrt(var), np.nan)


def _shrunk_corr(e: np.ndarray, min_obs: int) -> tuple[np.ndarray, float]:
    """Constant-correlation Ledoit-Wolf on residuals with gaps; PSD, unit diagonal."""
    t, n = e.shape
    cnt = np.sum(~np.isnan(e), axis=0)
    sd = _nanstd(e, ddof=0)
    usable = np.where((cnt >= min_obs) & (sd > 0))[0]
    out = np.eye(n)
    if len(usable) < 2:
        return out, 1.0
    x = e[:, usable]
    z = (x - np.nanmean(x, axis=0)) / sd[usable]
    z = np.nan_to_num(z, nan=0.0)  # missing bars carry no information about dependence
    z = z[np.any(~np.isnan(x), axis=1)]
    c, delta = ledoit_wolf_constant_corr(z)
    block = np.ix_(usable, usable)
    out[block] = c
    rest = np.setdiff1d(np.arange(n), usable)
    if len(rest):
        off = c[~np.eye(len(usable), dtype=bool)]
        rbar = float(off.mean()) if off.size else 0.0
        out[np.ix_(rest, np.arange(n))] = rbar
        out[np.ix_(np.arange(n), rest)] = rbar
    np.fill_diagonal(out, 1.0)
    return psd_corr(out), delta


def calibrate(
    prices: pd.DataFrame,
    spec: ScenarioSpec,
    *,
    clock: Clock,
    loading_overrides: Mapping[str, Mapping[str, float]] | None = None,
    source: FactorProxySource | None = None,
    factors: pd.DataFrame | None = None,
    calm_windows: Sequence[tuple[str, str]] | None = None,
    min_obs: int = 20,
    calm_drift: Literal["zero", "historical"] = "zero",
) -> Calibration:
    """Calibrate the generator on ``prices`` for the windows of ``spec``.

    Parameters
    ----------
    prices:
        Asset price levels, DatetimeIndex, one column per asset, at the bar
        frequency of ``clock``. Intraday prices (more than one bar per date)
        are accepted: the intraday profile is estimated, then prices are
        aggregated to daily (the last price of each session) and everything
        else is calibrated at the daily frequency. A daily panel with an
        intraday ``clock`` calibrates normally and stores a FLAT intraday
        profile (UserWarning).
    factors:
        Factor return panel (offline use). When omitted, ``source`` is used to
        build the factors named by the spec's anchors and conditional factors.
    calm_windows:
        Windows treated as the calm regime. Default: every bar outside the
        spec's stress windows. Passing explicit quiet periods is better when
        the history contains other crises.
    calm_drift:
        ``"zero"`` (default) removes the calm-window mean of every asset and
        factor column from the bootstrap pool (volatility, correlation and
        tail structure are kept); ``"historical"`` keeps the raw calm returns.
    """
    if calm_drift not in ("zero", "historical"):
        raise ValueError(f"calm_drift must be 'zero' or 'historical', got {calm_drift!r}")
    if factors is None and source is None:
        raise ValueError("calibrate needs `source` (a FactorProxySource) or a `factors` frame")
    if prices.shape[1] == 0:
        raise ValueError("prices has no asset columns")
    intraday: IntradayProfile | None = None
    if isinstance(prices.index, pd.DatetimeIndex) and bars_per_session(prices.index) > 1.5:
        pclock = Clock(calendar=prices.index)
        intraday = estimate_intraday_profile(prices.sort_index(), pclock)
        prices = aggregate_to_daily(prices.sort_index(), pclock)
        clock = pclock.daily()
    elif clock.is_intraday:
        slots = int(round(float(clock.bars_per_day)))
        warn_flat(slots)
        intraday = IntradayProfile.flat(slots, list(map(str, prices.columns)),
                                        has_overnight_gap=bool(clock.overnight_gap))
        clock = clock.daily()
    if not spec.calibration_windows:
        raise ValueError("the spec has no calibration_windows")
    prices = prices.sort_index().astype(float)
    if (prices <= 0).any().any():
        raise ValueError("prices must be positive")
    returns = np.log(prices).diff().iloc[1:]
    warns: list[str] = []

    needed = list(dict.fromkeys([*spec.anchors, *spec.conditional_factors]))
    if factors is None:
        order = [n for n in FACTOR_REGISTRY if n in needed] + [
            n for n in needed if n not in FACTOR_REGISTRY
        ]
        factors = build_factor_returns(
            source,
            prices.index[0].strftime("%Y-%m-%d"),
            prices.index[-1].strftime("%Y-%m-%d"),
            clock,
            factors=order,
            windows=list(spec.calibration_windows),
        )
        warns.extend(factors.attrs.get("warnings", []))
    missing = [n for n in needed if n not in factors.columns]
    if missing:
        raise ValueError(
            f"factor(s) {missing} named by the spec are not in the factor panel "
            f"(columns: {list(factors.columns)})"
        )

    idx = returns.index.intersection(factors.index).sort_values()
    if len(idx) < 3 * min_obs:
        raise ValueError("prices and factors share too few bars")
    R = returns.loc[idx]
    F = factors.loc[idx].astype(float)
    assets, fnames = tuple(R.columns), tuple(F.columns)

    stress_wins = [tuple(w) for w in spec.calibration_windows]
    smask = _window_mask(idx, stress_wins)
    if calm_windows is None:
        cmask = ~smask
    else:
        cmask = _window_mask(idx, calm_windows) & ~smask
    if smask.sum() < min_obs:
        raise ValueError("fewer than min_obs bars fall inside the calibration windows")
    if cmask.sum() < min_obs:
        raise ValueError("fewer than min_obs calm bars are available")
    calm_wins = _runs(idx, cmask)

    L = estimate_loadings(
        R, F, {"calm": calm_wins, "stress": stress_wins}, overrides=loading_overrides,
        min_obs=min_obs,
    )
    warns.extend(L.warnings)
    bc = np.nan_to_num(L.betas["calm"].loc[list(assets), list(fnames)].to_numpy(float))
    bs = np.nan_to_num(L.betas["stress"].loc[list(assets), list(fnames)].to_numpy(float))

    Rn, Fn = R.to_numpy(), F.to_numpy()
    e_calm = Rn[cmask] - Fn[cmask] @ bc.T
    e_stress = Rn[smask] - Fn[smask] @ bs.T
    sig_c = _nanstd(e_calm, ddof=1)
    sig_s = _nanstd(e_stress, ddof=1)
    ok = np.isfinite(sig_c) & (sig_c > 0)
    sig_c = np.where(ok, sig_c, np.nan)
    ratio = sig_s / sig_c
    good = np.isfinite(ratio) & (ratio > 0)
    fill = float(np.median(ratio[good])) if good.any() else 1.0
    if (~good).any():
        warns.append(
            "assets without a usable residual in a regime take the median stress/calm "
            f"residual ratio of the others ({fill:.2f}): "
            + ", ".join(np.array(assets)[~good])
        )
    ratio = np.where(good, ratio, fill)
    sig_c = np.where(np.isfinite(sig_c), sig_c, np.nanmedian(sig_c) if ok.any() else 0.01)

    corr_c, d_c = _shrunk_corr(e_calm, min_obs)
    corr_s, d_s = _shrunk_corr(e_stress, min_obs)

    caps = np.full(len(assets), np.nan)
    for a, b in stress_wins:
        sub = R.loc[pd.Timestamp(a) : pd.Timestamp(b)]
        roll = sub.rolling(CAP_WINDOW, min_periods=CAP_MIN_PERIODS).std().max()
        caps = np.fmax(caps, roll.to_numpy(float))
    no_cap = ~np.isfinite(caps)
    if no_cap.any():
        warns.append(
            "no volatility cap (not enough bars in the windows): "
            + ", ".join(np.array(assets)[no_cap])
        )
    caps = np.where(no_cap, np.inf, caps)

    fs, fc = Fn[smask], Fn[cmask]
    cov_s = np.atleast_2d(np.cov(fs, rowvar=False))
    cov_c = np.atleast_2d(np.cov(fc, rowvar=False))
    sig_fc = np.sqrt(np.diag(cov_c))
    vol_ratio = np.sqrt(np.diag(cov_s)) / np.where(sig_fc > 0, sig_fc, np.nan)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 1.0)

    anchored = [n for n in fnames if n in spec.anchors]
    m2_s = fs.T @ fs / len(fs)
    g, eta, cnames = conditional_map(m2_s, fnames, anchored)
    gamma = pd.DataFrame(g, index=cnames, columns=anchored)
    eta_df = pd.DataFrame(eta, index=cnames, columns=cnames)

    extra = [tuple(spec.analog_window)] if spec.analog_window else []
    windows = list(dict.fromkeys(stress_wins + extra))
    analog = {}
    for a, b in windows:
        analog[(a, b)] = F.loc[pd.Timestamp(a) : pd.Timestamp(b)].copy()

    pool_df = pd.concat([R, F], axis=1).loc[cmask]
    if calm_drift == "zero":
        pool_df = pool_df - pool_df.mean(skipna=True)  # own valid-bar mean per column
    pool = pool_df.fillna(0.0).to_numpy(float)
    return Calibration(
        assets=assets,
        factors=fnames,
        stress_windows=tuple(stress_wins),
        calm_windows=tuple(calm_wins),
        betas_calm=bc,
        betas_stress=bs,
        unaffected=tuple(a for a in assets if a in L.unaffected),
        resid_sigma_calm=sig_c,
        resid_ratio=ratio,
        resid_corr_calm=corr_c,
        resid_corr_stress=corr_s,
        vol_cap=caps,
        factor_mean_stress=fs.mean(axis=0),
        factor_cov_stress=cov_s,
        factor_m2_stress=m2_s,
        factor_cov_calm=cov_c,
        factor_sigma_calm=sig_fc,
        factor_vol_ratio=vol_ratio,
        gamma=gamma,
        eta_cov=eta_df,
        calm_pool=pool,
        analog_paths=analog,
        last_price=prices.iloc[-1].copy(),
        bars_per_year=float(clock.bars_per_year),
        shrinkage=(d_c, d_s),
        loadings=L,
        warnings=tuple(warns),
        source=None if source is None else type(source).__name__,
        calm_drift=calm_drift,
        intraday=intraday,
    )


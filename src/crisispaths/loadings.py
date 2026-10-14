"""Factor loading estimation for a user's own asset universe.

Model, per asset and regime::

    r_t = a + sum_f sum_{l in lags} b_{f,l} * F_{f,t-l} + e_t,   beta_f = sum_l b_{f,l}

The lead/lag sum (Dimson) handles assets whose closes are not synchronous with
the factor proxies. Coefficients are ridge-shrunk toward a prior with
``lambda = k * sigma2_resid / n_obs`` in standardised units. The calm prior is
zero; the stress prior is the calm coefficient vector.

Nothing here depends on asset-class labels: relationships between assets are
learned only from the user's own return panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

UNAFFECTED_TEXT = (
    "this asset is modelled as unaffected by the scenario, which is a model claim, not a fact"
)
_LAGS = (1, 0, -1)  # shift(1) is the lagged factor, shift(-1) the lead
# Imputed stress betas are the calm betas times ONE scalar per asset (never a per-factor
# ratio, which is unstable and can flip signs). The scalar is a through-origin least-squares
# fit over the neighbours, clipped to these bounds. The bounds are a judgment, not estimated.
AMPLIFICATION_MIN = 0.5
AMPLIFICATION_MAX = 2.0
_DENOM_FLOOR = 1e-4  # sum of squared neighbour calm betas below which s falls back to 1.0


def stress_amplification(calm: np.ndarray, stress: np.ndarray) -> tuple[float, bool, bool]:
    """Pooled through-origin amplification ``s = sum(stress * calm) / sum(calm ** 2)``.

    ``calm`` and ``stress`` are (neighbours x factors) beta arrays. Returns
    ``(s, clipped, degenerate)``: ``s`` clipped to ``[AMPLIFICATION_MIN,
    AMPLIFICATION_MAX]``; when the pooled denominator is about zero ``s = 1.0``
    and ``degenerate`` is True.
    """
    cm, sm = np.asarray(calm, float), np.asarray(stress, float)
    den = float(np.sum(cm * cm))
    if not den > _DENOM_FLOOR:
        return 1.0, False, True
    raw = float(np.sum(sm * cm)) / den
    s = float(np.clip(raw, AMPLIFICATION_MIN, AMPLIFICATION_MAX))
    return s, s != raw, False


@dataclass
class Loadings:
    """Estimated loadings. ``betas[regime]`` is an assets x factors frame."""

    betas: dict[str, pd.DataFrame]
    std_errors: dict[str, pd.DataFrame]
    n_obs: pd.DataFrame
    shrinkage: pd.DataFrame
    r2: pd.DataFrame
    unaffected: set[str] = field(default_factory=set)
    imputed: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Fit:
    coef: np.ndarray  # full coefficient vector, original units
    cov: np.ndarray
    n: int
    r2: float
    share: float


def _regime_mask(index: pd.DatetimeIndex, wins: Sequence[tuple]) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for a, b in wins:
        mask |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return mask


def _ridge(y: np.ndarray, X: np.ndarray, prior: np.ndarray, k: float) -> _Fit | None:
    n, p = X.shape
    if n < 2 * p + 2:
        return None
    xm, ym = X.mean(axis=0), y.mean()
    sx = X.std(axis=0)
    sx = np.where(sx > 0, sx, 1.0)
    sy = y.std()
    sy = sy if sy > 0 else 1.0
    Z = (X - xm) / sx
    u = (y - ym) / sy
    S = Z.T @ Z / n
    c = Z.T @ u / n
    g_ols = np.linalg.lstsq(Z, u, rcond=None)[0]
    s2 = float(np.sum((u - Z @ g_ols) ** 2)) / max(n - p - 1, 1)
    lam = k * s2 / n
    g0 = prior * sx / sy
    A = S + lam * np.eye(p)
    Ainv = np.linalg.pinv(A)
    g = Ainv @ (c + lam * g0)
    resid = u - Z @ g
    s2r = float(resid @ resid) / max(n - p - 1, 1)
    cov_g = s2r / n * Ainv @ S @ Ainv
    scale = sy / sx
    coef = g * scale
    cov = cov_g * np.outer(scale, scale)
    tss = float(u @ u)
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else 0.0
    return _Fit(coef, cov, n, r2, lam / (1.0 + lam))


def _exposure_test(y: np.ndarray, X: np.ndarray) -> tuple[float, float] | None:
    """Joint F-test p-value and adjusted R2 of an OLS regression with intercept."""
    n, p = X.shape
    if n < p + 3:
        return None
    D = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    rss = float(np.sum((y - D @ beta) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0:
        return 1.0, 0.0
    r2 = 1.0 - rss / tss
    dof = n - p - 1
    if rss <= 0:
        return 0.0, 1.0
    F = (r2 / p) / ((1.0 - r2) / dof)
    pval = float(stats.f.sf(F, p, dof))
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    return pval, adj


def estimate_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    windows: Mapping[str, Sequence[tuple]],
    regimes: Sequence[str] = ("calm", "stress"),
    overrides: Mapping[str, Mapping[str, float]] | None = None,
    *,
    dimson: bool = True,
    k: float = 1.0,
    n_neighbors: int = 3,
    min_obs: int = 60,
    exposure_p: float = 0.10,
    exposure_r2: float = 0.02,
) -> Loadings:
    """Estimate calm and stress factor loadings for every column of ``returns``.

    Parameters
    ----------
    returns:
        Asset returns (one column per asset), DatetimeIndex.
    factors:
        Factor returns (one column per factor), same frequency.
    windows:
        ``{regime: [(start, end), ...]}``, with one entry per name in ``regimes``.
    regimes:
        ``(calm, stress)``. The calm prior is zero; the stress prior is the calm
        coefficient vector.
    overrides:
        ``{asset: {factor: beta}}``. Applied last and recorded in ``metadata``.
    dimson:
        Use lead/lag factor columns (lags -1, 0, +1) and report their sum.
    k:
        Ridge strength in ``lambda = k * sigma2_resid / n_obs``.
    n_neighbors:
        Neighbours used when an asset has no data in the stress windows.
    min_obs:
        Warn when a regime fit uses fewer observations than this.
    exposure_p, exposure_r2:
        No-exposure rule: pooled joint F-test p-value above ``exposure_p`` and
        adjusted R2 below ``exposure_r2`` zero the loadings and flag the asset.
        Adjusted R2 is used because the Dimson columns inflate raw R2.
    """
    if returns.shape[1] == 0:
        raise ValueError("returns has no asset columns")
    if len(regimes) != 2:
        raise ValueError("regimes must be a (calm, stress) pair")
    missing = [r for r in regimes if r not in windows]
    if missing:
        raise ValueError(f"windows is missing regime(s): {missing}")
    assets, fnames = list(returns.columns), list(factors.columns)
    overrides = {a: dict(v) for a, v in (overrides or {}).items()}
    for a, fv in overrides.items():
        if a not in assets:
            raise KeyError(f"override for unknown asset {a!r}")
        for f in fv:
            if f not in fnames:
                raise KeyError(f"override for unknown factor {f!r} (asset {a!r})")

    idx = returns.index.intersection(factors.index).sort_values()
    R = returns.loc[idx].astype(float)
    F = factors.loc[idx].astype(float)
    lags = _LAGS if dimson else (0,)
    blocks = [F.shift(l) for l in lags]  # noqa: E741
    X_all = np.column_stack([b.to_numpy() for b in blocks])
    nf, nl = len(fnames), len(lags)
    sum_map = np.zeros((nf, nf * nl))
    for li in range(nl):
        for j in range(nf):
            sum_map[j, li * nf + j] = 1.0
    masks = {r: _regime_mask(idx, windows[r]) for r in regimes}
    pooled = masks[regimes[0]] | masks[regimes[1]]
    calm, stress = regimes

    betas = {r: pd.DataFrame(np.nan, index=assets, columns=fnames) for r in regimes}
    ses = {r: pd.DataFrame(np.nan, index=assets, columns=fnames) for r in regimes}
    n_obs = pd.DataFrame(0, index=assets, columns=list(regimes))
    shrink = pd.DataFrame(np.nan, index=assets, columns=list(regimes))
    r2 = pd.DataFrame(np.nan, index=assets, columns=list(regimes))
    warns: list[str] = []
    unaffected: set[str] = set()
    imputed: set[str] = set()
    fits: dict[tuple[str, str], _Fit] = {}
    imputation: dict[str, dict] = {}

    def valid(y: np.ndarray, m: np.ndarray) -> np.ndarray:
        return m & ~np.isnan(y) & ~np.isnan(X_all).any(axis=1)

    for a in assets:
        y = R[a].to_numpy()
        prior = np.zeros(X_all.shape[1])
        for r in regimes:
            ok = valid(y, masks[r])
            fit = _ridge(y[ok], X_all[ok], prior, k)
            n_obs.loc[a, r] = int(ok.sum())
            if fit is None:
                continue
            fits[(a, r)] = fit
            if r == calm:
                prior = fit.coef
            betas[r].loc[a] = sum_map @ fit.coef
            cov_sum = sum_map @ fit.cov @ sum_map.T
            ses[r].loc[a] = np.sqrt(np.clip(np.diag(cov_sum), 0.0, None))
            shrink.loc[a, r] = fit.share
            r2.loc[a, r] = fit.r2
            if fit.n < min_obs:
                warns.append(
                    f"{a}: n_obs={fit.n} < {min_obs} in regime '{r}'; estimates are unreliable"
                )
        ok = valid(y, pooled)
        test = _exposure_test(y[ok], X_all[ok])
        if test is not None and test[0] > exposure_p and test[1] < exposure_r2:
            unaffected.add(a)
            for r in regimes:
                betas[r].loc[a] = 0.0
            warns.append(f"{a}: {UNAFFECTED_TEXT}")

    # assets with a calm fit but no usable stress fit: borrow from nearest neighbours
    donors = [
        a for a in assets
        if (a, calm) in fits and (a, stress) in fits and a not in unaffected
    ]
    for a in assets:
        if a in unaffected or (a, stress) in fits:
            continue
        if (a, calm) not in fits:
            for r in regimes:
                betas[r].loc[a] = 0.0
            warns.append(f"{a}: no usable data in any regime; loadings set to zero")
            continue
        c = betas[calm].loc[a].to_numpy()
        others = [d for d in donors if d != a]
        order = sorted(
            others, key=lambda d: (float(np.linalg.norm(betas[calm].loc[d].to_numpy() - c)), d)
        )[: max(n_neighbors, 1)]
        ratio = 1.0
        info: dict[str, Any] = {"neighbours": list(order), "amplification": 1.0,
                                "clipped": False, "degenerate": False}
        if order:
            cm = np.vstack([betas[calm].loc[d].to_numpy() for d in order])
            sm = np.vstack([betas[stress].loc[d].to_numpy() for d in order])
            ratio, clipped, degenerate = stress_amplification(cm, sm)
            info.update(amplification=ratio, clipped=clipped, degenerate=degenerate)
            how = (
                "the neighbours' calm loadings are about zero, so amplification s=1.00 "
                "(no amplification information)" if degenerate else
                f"amplification s={ratio:.2f}"
                + (f" (clipped to [{AMPLIFICATION_MIN}, {AMPLIFICATION_MAX}], a judgment "
                   "bound)" if clipped else "")
            )
            warns.append(
                f"{a}: no usable data in stress windows; stress beta = calm beta * one scalar, "
                f"{how}, a pooled least-squares fit over its {len(order)} nearest assets in "
                f"calm-loading space ({', '.join(order)})"
            )
        else:
            warns.append(
                f"{a}: no usable data in stress windows and no neighbours available; "
                "stress beta set equal to calm beta (s=1.00)"
            )
        imputation[a] = info
        betas[stress].loc[a] = c * ratio
        imputed.add(a)

    for a, fv in overrides.items():
        for f, v in fv.items():
            for r in regimes:
                betas[r].loc[a, f] = float(v)
                ses[r].loc[a, f] = np.nan
        if a in unaffected:
            unaffected.discard(a)
            warns.append(f"{a}: manual override supplied; asset no longer flagged unaffected")

    meta = {
        "regimes": list(regimes),
        "windows": {r: [(str(s), str(e)) for s, e in windows[r]] for r in regimes},
        "dimson_lags": [-l for l in lags] if dimson else [0],  # noqa: E741
        "k": k,
        "n_neighbors": n_neighbors,
        "min_obs": min_obs,
        "exposure_p": exposure_p,
        "exposure_r2": exposure_r2,
        "overrides": overrides,
        "imputation": imputation,
    }
    return Loadings(
        betas=betas,
        std_errors=ses,
        n_obs=n_obs,
        shrinkage=shrink,
        r2=r2,
        unaffected=unaffected,
        imputed=imputed,
        warnings=warns,
        metadata=meta,
    )

"""Path metrics, benchmark-relative metrics and across-path summaries.

Conventions
-----------
* ``equity_sim`` includes the start point (``iloc[0]``): ``n + 1`` points for
  ``n`` simulated bars. All times are in bars.
* Drawdowns are positive fractions (0.30 is a 30% fall from a peak).
* ``max_drawdown_from_start``: worst fall from the running peak of the
  simulated segment (its first point is the start equity).
  ``max_drawdown_from_peak``: worst fall from the all-time peak, i.e. the
  running peak of the segment, but never below ``history_peak`` when given.
  ``loss_from_start``: ``max(0, 1 - min(equity) / start)``.
* Time to trough and time to recover refer to the worst drawdown from the
  all-time peak. ``time_to_trough_bars`` counts from the segment start; the
  recovery clock starts at the trough and stops when equity regains the peak
  that drawdown fell from. A path that never does is CENSORED at the horizon:
  ``recovered=False``, ``time_to_recover_bars=NaN`` and ``recovery_time_bars``
  holds the bars observed after the trough (the censoring time).
* A path with no drawdown at all counts as recovered at time 0.
* Across paths, recovery is summarised with Kaplan-Meier, never with a mean of
  the observed times (those are right-censored).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_PATHS_FOR_CIS = 200
CI_LEVEL = 0.95
LOG_FLOOR = 1e-12
_CHUNK = 250
REQUIRED_COLUMNS = (
    "terminal_wealth",
    "max_drawdown_from_peak",
    "ruined",
    "recovered",
    "recovery_time_bars",
)


# --- single path -------------------------------------------------------------------------


def path_metrics(
    equity_sim: pd.Series,
    *,
    history_peak: float | None,
    ruin_level: float = 0.5,
    horizon_bars: int | None = None,
) -> dict:
    """Metrics of one simulated equity curve (see the module docstring for conventions).

    ``history_peak`` is the all-time peak of the history in the same units as
    ``equity_sim`` (``None``: no history peak). ``ruin_level`` is the fraction
    of start equity at or below which the path counts as ruined.
    ``horizon_bars`` limits the observation window (default: the whole curve).
    """
    e = np.asarray(equity_sim, dtype=float)
    if e.ndim != 1 or len(e) < 1:
        raise ValueError("equity_sim must be a non-empty 1-D series")
    if not np.isfinite(e).all():
        raise ValueError("equity_sim must be finite (it contains NaN or inf)")
    start = e[0]
    if not start > 0:
        raise ValueError(f"start equity must be positive, got {start}")
    full = len(e) - 1
    h = full if horizon_bars is None else int(min(max(horizon_bars, 0), full))
    e = e[: h + 1]

    run = np.maximum.accumulate(e)
    ref = run if history_peak is None else np.maximum(run, max(float(history_peak), start))
    dd_start = 1.0 - e / run
    dd_peak = 1.0 - e / ref
    trough = int(np.argmax(dd_peak))
    max_peak = float(dd_peak[trough])
    if max_peak <= 0.0:
        trough, recovered, t_rec, t_obs = 0, True, 0, 0
    else:
        level = ref[trough]
        after = np.where(e[trough + 1:] >= level)[0]
        recovered = bool(after.size)
        t_rec = int(after[0]) + 1 if recovered else None
        t_obs = t_rec if recovered else h - trough
    ruin_hit = np.where(e <= ruin_level * start)[0]
    tw = float(e[-1] / start)
    return {
        "terminal_wealth": tw,
        "terminal_log_return": math.log(max(tw, LOG_FLOOR)),
        "max_drawdown_from_start": float(dd_start.max()),
        "max_drawdown_from_peak": max_peak,
        "loss_from_start": float(max(0.0, 1.0 - e.min() / start)),
        "time_to_trough_bars": trough,
        "time_to_recover_bars": float(t_rec) if recovered else float("nan"),
        "recovered": recovered,
        "recovery_time_bars": float(t_obs),
        "ruined": bool(ruin_hit.size),
        "time_to_ruin_bars": float(ruin_hit[0]) if ruin_hit.size else float("nan"),
        "horizon_bars": h,
    }


def relative_metrics(strategy: dict, benchmark: dict) -> dict:
    """Strategy minus benchmark: excess terminal wealth and the drawdown difference.

    Drawdown difference is positive when the strategy suffered the deeper fall.
    """
    return {
        "excess_terminal_wealth": strategy["terminal_wealth"] - benchmark["terminal_wealth"],
        "excess_max_drawdown_from_peak": strategy["max_drawdown_from_peak"]
        - benchmark["max_drawdown_from_peak"],
    }


# --- estimators ----------------------------------------------------------------------------


def kaplan_meier(times, events) -> pd.DataFrame:
    """Kaplan-Meier survival of "not yet recovered".

    ``events[i]`` is True when ``times[i]`` is an observed recovery, False when
    it is a censoring time. Returns a frame ``time, at_risk, events, survival``
    with ``attrs`` ``share_recovered`` (1 - S at the last observed time) and
    ``median_time`` (first time with S <= 0.5, NaN if never reached).
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(events, dtype=bool)
    if t.shape != d.shape or t.ndim != 1 or len(t) == 0:
        raise ValueError("times and events must be 1-D arrays of the same non-zero length")
    uniq, inv = np.unique(t, return_inverse=True)
    count = np.bincount(inv, minlength=len(uniq))
    n_ev = np.bincount(inv, weights=d.astype(float), minlength=len(uniq)).astype(int)
    n_at = len(t) - (np.cumsum(count) - count)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.cumprod(1.0 - n_ev / n_at)
    out = pd.DataFrame({"time": uniq, "at_risk": n_at, "events": n_ev, "survival": s})
    below = np.where(s <= 0.5 + 1e-12)[0]
    out.attrs["share_recovered"] = float(1.0 - s[-1])
    out.attrs["median_time"] = float(uniq[below[0]]) if below.size else float("nan")
    return out


def cvar(x, alpha: float = 0.05, *, tail: str = "low") -> float:
    """Mean of the worst ``ceil(alpha * n)`` values (``low``: smallest, ``high``: largest)."""
    v = np.sort(np.asarray(x, dtype=float))
    if v.size == 0:
        return float("nan")
    k = max(1, math.ceil(alpha * v.size - 1e-12))
    return float(v[:k].mean() if tail == "low" else v[-k:].mean())


def _cvar_rows(s: np.ndarray, alpha: float, tail: str) -> np.ndarray:
    k = max(1, math.ceil(alpha * s.shape[1] - 1e-12))
    srt = np.sort(s, axis=1)
    return srt[:, :k].mean(axis=1) if tail == "low" else srt[:, -k:].mean(axis=1)


def wilson_interval(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + level / 2.0))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def bootstrap_ci(
    x,
    stat: Callable[[np.ndarray], np.ndarray],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = CI_LEVEL,
    return_replicates: bool = False,
):
    """Percentile bootstrap across paths (resampling rows with replacement).

    ``stat`` maps a ``(b, n)`` array of resamples to ``(b,)`` or ``(b, m)``
    statistics. Returns ``(low, high)`` (arrays when ``m > 1``) or, with
    ``return_replicates``, ``(low, high, replicates)``. Deterministic in ``seed``.
    """
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("bootstrap needs a non-empty 1-D sample")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    reps = []
    done = 0
    while done < n_boot:
        b = min(_CHUNK, n_boot - done)
        idx = rng.integers(0, v.size, size=(b, v.size))
        reps.append(np.asarray(stat(v[idx]), dtype=float))
        done += b
    r = np.concatenate(reps, axis=0)
    a = (1.0 - level) / 2.0
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(r, [100 * a, 100 * (1 - a)], axis=0)
    return (lo, hi, r) if return_replicates else (lo, hi)


def _km_boot(times: np.ndarray, events: np.ndarray, n_boot: int, seed: int, level: float):
    rng = np.random.default_rng(seed)
    n = len(times)
    share = np.empty(n_boot)
    med = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        km = kaplan_meier(times[i], events[i])
        share[b] = km.attrs["share_recovered"]
        med[b] = km.attrs["median_time"]
    a = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    with np.errstate(all="ignore"):
        s_ci = tuple(np.nanpercentile(share, a))
        m_ci = tuple(np.nanpercentile(med, a)) if np.isfinite(med).any() else (np.nan, np.nan)
    return s_ci, m_ci, float(np.isnan(med).mean())


# --- summary -------------------------------------------------------------------------------


@dataclass
class Summary:
    """Across-path summary. All intervals are percentile-bootstrap over paths, except
    the ruin probability (Wilson)."""

    n_paths: int
    alpha: float
    quantiles: pd.DataFrame  # metric, quantile, value, ci_low, ci_high
    cvar: dict  # metric -> {tail, alpha, value, ci_low, ci_high}
    ruin: dict  # p, n_ruined, n, ci_low, ci_high
    recovery: dict  # share_recovered, median_time (+ CIs), n_censored, horizon_bars, km
    caveats: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def summarize(
    results_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    analog_exact: bool = False,
    ci_level: float = CI_LEVEL,
) -> Summary:
    """Summarise a one-row-per-path results table (see ``backtest``)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df is missing column(s) {missing}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    df = results_df
    n = len(df)
    if n == 0:
        raise ValueError("results_df has no rows")

    qs = np.array(QUANTILES)
    metrics = ["terminal_wealth", "max_drawdown_from_peak"]
    if "excess_terminal_wealth" in df.columns:
        metrics.append("excess_terminal_wealth")
    rows = []
    for m in metrics:
        x = df[m].to_numpy(float)
        lo, hi = bootstrap_ci(
            x, lambda s: np.quantile(s, qs, axis=1).T, n_boot=n_boot, seed=seed, level=ci_level
        )
        val = np.quantile(x, qs)
        for j, q in enumerate(QUANTILES):
            rows.append({"metric": m, "quantile": q, "value": float(val[j]),
                         "ci_low": float(lo[j]), "ci_high": float(hi[j])})
    quant = pd.DataFrame(rows)

    cv = {}
    for m, tail in (("terminal_wealth", "low"), ("max_drawdown_from_peak", "high")):
        x = df[m].to_numpy(float)
        lo, hi = bootstrap_ci(
            x, lambda s, t=tail: _cvar_rows(s, alpha, t), n_boot=n_boot, seed=seed,
            level=ci_level,
        )
        cv[m] = {"tail": tail, "alpha": alpha, "value": cvar(x, alpha, tail=tail),
                 "ci_low": float(lo), "ci_high": float(hi)}

    k = int(df["ruined"].sum())
    wl, wh = wilson_interval(k, n, ci_level)
    ruin = {"p": k / n, "n_ruined": k, "n": n, "ci_low": wl, "ci_high": wh}

    times = df["recovery_time_bars"].to_numpy(float)
    ev = df["recovered"].to_numpy(bool)
    km = kaplan_meier(times, ev)
    s_ci, m_ci, m_nan = _km_boot(times, ev, n_boot, seed, ci_level)
    n_cens = int((~ev).sum())
    horizon = float(df["horizon_bars"].iloc[0]) if "horizon_bars" in df.columns else float("nan")
    recovery = {
        "share_recovered": km.attrs["share_recovered"],
        "share_recovered_ci": s_ci,
        "median_time": km.attrs["median_time"],
        "median_time_ci": m_ci,
        "median_undefined_share_of_resamples": m_nan,
        "n_censored": n_cens,
        "n": n,
        "horizon_bars": horizon,
        "km": km,
    }

    caveats: list[str] = []
    if n < MIN_PATHS_FOR_CIS:
        caveats.append(
            f"only {n} paths (< {MIN_PATHS_FOR_CIS}): quantile, CVaR and recovery intervals are "
            "unstable; use at least 200 paths for tail statistics"
        )
    if analog_exact:
        caveats.append(
            "analog exact mode: the crash replays one historical window in every path, so the "
            "intervals are conditional on that single history and understate scenario uncertainty"
        )
    if n_cens:
        caveats.append(
            f"{n_cens} of {n} paths did not recover within the horizon (censored); recovery is "
            "estimated with Kaplan-Meier, the median is undefined when fewer than half recover"
        )
    return Summary(
        n_paths=n,
        alpha=alpha,
        quantiles=quant,
        cvar=cv,
        ruin=ruin,
        recovery=recovery,
        caveats=caveats,
        meta={"n_boot": n_boot, "seed": seed, "ci_level": ci_level, "alpha": alpha},
    )

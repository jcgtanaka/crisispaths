"""One-call backtest: scenario -> calibrate -> generate -> run -> summarise.

Single process. For long, parallel or resumable runs use
:func:`crisispaths.runner.run_mc`; this module holds the reference semantics
(:func:`run_one_path`) that the runner reuses, so both give identical rows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibrate import Calibration, calibrate
from .clock import Clock
from .compose import compose
from .generate import generate_path
from .harness import StrategyResult, buy_and_hold_equal_weight, run_strategy
from .library import ALIASES, load_scenario
from .lookahead import LookaheadError
from .lookahead import check_lookahead as _audit
from .metrics import Summary, path_metrics, relative_metrics, summarize
from .spec import ScenarioSpec

MODES = ("weights", "returns")


def calibration_hash(cal: Calibration) -> str:
    """SHA-256 of the numeric content of a calibration (stable across processes)."""
    h = hashlib.sha256()

    def add(x) -> None:
        if isinstance(x, pd.DataFrame):
            h.update(repr((list(map(str, x.index)), list(map(str, x.columns)))).encode())
            h.update(np.ascontiguousarray(x.to_numpy(float)).tobytes())
        elif isinstance(x, pd.Series):
            h.update(repr(list(map(str, x.index))).encode())
            h.update(np.ascontiguousarray(x.to_numpy(float)).tobytes())
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x, dtype=float).tobytes())
        else:
            h.update(repr(x).encode())

    for name in (
        "assets", "factors", "stress_windows", "calm_windows", "unaffected", "bars_per_year",
        "betas_calm", "betas_stress", "resid_sigma_calm", "resid_ratio", "resid_corr_calm",
        "resid_corr_stress", "vol_cap", "factor_mean_stress", "factor_cov_stress",
        "factor_m2_stress", "factor_cov_calm", "factor_sigma_calm", "factor_vol_ratio",
        "gamma", "eta_cov", "calm_pool", "last_price", "calm_drift",
    ):
        h.update(name.encode())
        add(getattr(cal, name))
    if cal.intraday is not None:  # only hashed when present: daily hashes are unchanged
        p = cal.intraday
        h.update(b"intraday")
        for name in ("weights", "pooled", "raw_weights", "n_sessions", "shrinkage",
                     "overnight_share", "daily_vol"):
            add(getattr(p, name))
        h.update(repr((p.assets, p.slots, p.has_overnight_gap, p.df, p.source,
                       p.slot_times)).encode())
    for key in sorted(cal.analog_paths):
        h.update(repr(key).encode())
        add(cal.analog_paths[key])
    return h.hexdigest()


def _resolve_spec(scenario) -> ScenarioSpec:
    if isinstance(scenario, ScenarioSpec):
        return scenario
    if isinstance(scenario, str):
        return load_scenario(scenario) if scenario in ALIASES else compose(scenario)
    raise TypeError(
        f"scenario must be a ScenarioSpec or a name (alias or family), got {type(scenario)}"
    )


def _uses_exact_analog(spec: ScenarioSpec) -> bool:
    analog = any(a.cum_dist == "analog" for a in spec.anchors.values())
    return spec.analog_mode == "exact" and (analog or spec.aftermath.mode == "analog")


def resolve_spec(scenario) -> ScenarioSpec:
    """A ``ScenarioSpec``, an alias name or a family name, as a spec."""
    return _resolve_spec(scenario)


def uses_exact_analog(spec: ScenarioSpec) -> bool:
    """True when the scenario replays one historical window exactly in every path."""
    return _uses_exact_analog(spec)


def prepare_run(
    prices: pd.DataFrame,
    spec: ScenarioSpec,
    *,
    clock: Clock | None = None,
    calibration: Calibration | None = None,
    factors: pd.DataFrame | None = None,
    source=None,
    calm_drift: str | None = None,
) -> tuple[Clock, Calibration, str, pd.DataFrame]:
    """Resolve the clock, calibrate once and return ``(clock, cal, cal_hash, history)``."""
    if clock is None:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("pass clock= when prices does not have a DatetimeIndex")
        clock = Clock(calendar=prices.index)
    cal = calibration
    if cal is None:
        kw = {} if calm_drift is None else {"calm_drift": calm_drift}
        cal = calibrate(prices, spec, clock=clock, factors=factors, source=source, **kw)
    elif calm_drift is not None and calm_drift != cal.calm_drift:
        raise ValueError(
            f"calm_drift={calm_drift!r} conflicts with the given calibration "
            f"(calm_drift={cal.calm_drift!r}); recalibrate or drop the argument"
        )
    return clock, cal, calibration_hash(cal), prices[list(cal.assets)]


def run_on_path(
    strategy: Callable,
    history: pd.DataFrame,
    path,
    i: int,
    clock: Clock,
    *,
    mode: str = "weights",
    benchmark: Callable | None = buy_and_hold_equal_weight,
    ruin_level: float = 0.5,
    **run_kwargs,
) -> tuple[dict, pd.Series]:
    """Run the strategy (and benchmark) on an already generated path; see ``run_one_path``."""
    common = dict(clock=clock, **run_kwargs)
    res: StrategyResult = run_strategy(strategy, history, path, mode=mode, **common)
    hp = res.meta["history_peak"]
    m = path_metrics(res.equity, history_peak=hp, ruin_level=ruin_level)
    row = {"path": i, **m, "total_turnover": float(np.nansum(res.turnover.to_numpy())),
           "total_cost": res.meta["total_cost"], "onset_bars": path.meta["onset_bars"]}
    if benchmark is not None:
        b = run_strategy(benchmark, history, path, mode="weights", **common)
        bm = path_metrics(b.equity, history_peak=b.meta["history_peak"], ruin_level=ruin_level)
        row.update(
            bench_terminal_wealth=bm["terminal_wealth"],
            bench_max_drawdown_from_peak=bm["max_drawdown_from_peak"],
            bench_ruined=bm["ruined"],
            **relative_metrics(m, bm),
        )
    if clock.is_intraday:  # bars are intraday bars; also report calendar sessions
        bpd = float(clock.bars_per_day)
        row["time_to_trough_days"] = row["time_to_trough_bars"] / bpd
        row["time_to_recover_days"] = row["time_to_recover_bars"] / bpd
        row["recovery_time_days"] = row["recovery_time_bars"] / bpd
    return row, res.equity


def run_one_path(
    strategy: Callable,
    history: pd.DataFrame,
    cal: Calibration,
    spec: ScenarioSpec,
    i: int,
    seed: int,
    clock: Clock,
    *,
    dial: float = 1.0,
    mode: str = "weights",
    benchmark: Callable | None = buy_and_hold_equal_weight,
    ruin_level: float = 0.5,
    **run_kwargs,
) -> tuple[dict, pd.Series]:
    """Generate path ``i`` of ``seed``, run the strategy (and benchmark) on it.

    Returns ``(row, equity)``: the one-row metrics dict and the strategy equity
    curve of the simulated segment (start point included). A pure function of
    its arguments, so path ``i`` can be regenerated alone.
    """
    path = generate_path(cal, spec, i, seed, clock, dial=dial)
    return run_on_path(strategy, history, path, i, clock, mode=mode, benchmark=benchmark,
                       ruin_level=ruin_level, **run_kwargs)


@dataclass
class BacktestResult:
    """``table`` has one row per path; ``summary`` aggregates it across paths."""

    table: pd.DataFrame
    summary: Summary
    spec: ScenarioSpec
    calibration_hash: str
    meta: dict = field(default_factory=dict)


def backtest(
    strategy: Callable,
    prices: pd.DataFrame,
    scenario,
    *,
    n_paths: int = 200,
    seed: int = 0,
    clock: Clock | None = None,
    costs_bps: float = 0.0,
    dial: float = 1.0,
    calibration: Calibration | None = None,
    factors: pd.DataFrame | None = None,
    source=None,
    calm_drift: str | None = None,
    mode: str = "weights",
    benchmark: Callable | None = buy_and_hold_equal_weight,
    ruin_level: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = 2000,
    check_lookahead: bool = True,
    on_lookahead: str = "raise",
    **run_kwargs,
) -> BacktestResult:
    """Backtest ``strategy`` on ``n_paths`` paths of ``scenario`` (a name or a spec).

    ``prices`` is the user's real history (also the calibration panel). Path
    ``i`` uses ``SeedSequence([seed, i])``, so results are reproducible and
    independent of ``n_paths``. The benchmark (default: equal-weight, same
    harness, same path and costs) feeds the relative columns; pass
    ``benchmark=None`` to skip it. ``run_kwargs`` go to
    :func:`crisispaths.harness.run_strategy` (``leverage_cap``, ``cash_return``, ...).
    The look-ahead check runs once on the history, before any path.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    spec = _resolve_spec(scenario)
    clock, cal, cal_hash, history = prepare_run(
        prices, spec, clock=clock, calibration=calibration, factors=factors, source=source,
        calm_drift=calm_drift,
    )

    if check_lookahead and mode == "weights":
        rep = _audit(strategy, history)
        if not rep.passed:
            if on_lookahead == "raise":
                raise LookaheadError(rep.message())
            import warnings

            warnings.warn(rep.message(), UserWarning, stacklevel=2)

    rows = []
    for i in range(n_paths):
        row, _ = run_one_path(
            strategy, history, cal, spec, i, seed, clock, dial=dial, mode=mode,
            benchmark=benchmark, ruin_level=ruin_level, costs_bps=costs_bps, **run_kwargs,
        )
        rows.append(row)
    table = pd.DataFrame(rows)

    summary = summarize(table, alpha=alpha, n_boot=n_boot, seed=seed,
                        analog_exact=_uses_exact_analog(spec))
    from . import __version__

    meta = {
        "seed": int(seed),
        "n_paths": int(n_paths),
        "bars": int(cal_bars(table, spec, clock)),
        "dial": float(dial),
        "costs_bps": float(costs_bps),
        "mode": mode,
        "scenario": spec.key,
        "spec_content_hash": spec.content_hash,
        "spec_metadata": spec.metadata,
        "calibration_hash": cal_hash,
        "calibration_source": cal.source,
        "calibration_warnings": list(cal.warnings),
        "calm_drift": cal.calm_drift,
        "package_version": __version__,
        "benchmark": None if benchmark is None else getattr(benchmark, "__name__", "custom"),
        "ruin_level": float(ruin_level),
        "analog_exact": _uses_exact_analog(spec),
    }
    return BacktestResult(table=table, summary=summary, spec=spec, calibration_hash=cal_hash,
                          meta=meta)


def cal_bars(table: pd.DataFrame, spec: ScenarioSpec, clock: Clock) -> int:
    """Simulated bars per path (the observation horizon of the metrics)."""
    return int(table["horizon_bars"].iloc[0])

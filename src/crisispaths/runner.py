"""Resumable, parallel Monte Carlo runner.

``run_mc`` runs one or several scenarios and keeps everything on disk so that a
killed run resumes exactly where it stopped.

Files written to ``out_dir`` (``<key>`` is ``spec.key``)
--------------------------------------------------------
``results_<key>.csv``
    One row per path, appended with ``O_APPEND`` and ``fsync`` per row, so a
    kill loses at most the row being written. The first line is the header.
    Besides the metrics (the columns of ``backtest(...).table``) every row
    carries ``eq_00 .. eq_40``: the strategy equity relative to the start on 41
    evenly spaced bars (used by the report plots; dropped from the returned
    table).
``pin_<key>.json``
    Identity of the run: spec content hash, calibration hash, seed, package
    version and the run parameters that change results (costs, dial, mode,
    benchmark, ...). A resume with a different pin raises
    :class:`PinMismatchError`; runs are never silently mixed. The number of
    paths is not part of the pin: path ``i`` does not depend on it.
``manifest_<key>.json``
    Descriptive data for the report (spec, loadings, caveats). Rewritten on
    every run, never compared.

Path ``i`` uses ``SeedSequence([seed, i])`` (see :func:`crisispaths.generate_path`),
so any path regenerates alone and the rows are the same whether they were made
in one process, by several workers or across a kill and a resume. Rows are
written in path order, so the CSV bytes do not depend on ``workers`` either.

Workers use the ``spawn`` start method on every OS. A strategy is therefore
given as an importable ``"module:fn"`` string (or a top-level function of an
importable module, which is converted to that string).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import signal
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import (
    MODES,
    prepare_run,
    resolve_spec,
    run_on_path,
    uses_exact_analog,
)
from .calibrate import Calibration
from .clock import Clock
from .compose import enumerate_scenarios, grid_report
from .generate import generate_path
from .harness import buy_and_hold_equal_weight
from .lookahead import LookaheadError
from .lookahead import check_lookahead as _audit
from .spec import ScenarioSpec, _to_dict

EQ_POINTS = 41
EQ_PREFIX = "eq_"
# per-path diagnostics stored next to the metrics (used by the report's path sanity section)
CRASH_PREFIX = "crashret_"  # asset simple return over the crash leg (phase 1)
HORIZON_PREFIX = "horret_"  # asset simple return over the whole horizon
FACTOR_PREFIX = "fcrash_"  # factor cumulative return over the crash leg
EXTRA_PREFIXES = (EQ_PREFIX, CRASH_PREFIX, HORIZON_PREFIX, FACTOR_PREFIX)
DEFAULT_MAX_SCENARIOS = 64


class PinMismatchError(ValueError):
    """The results on disk were produced by a different run (spec, seed, calibration, ...)."""


# --- strategy resolution -----------------------------------------------------------------------


def _import_strategy(ref: str) -> Callable:
    if not isinstance(ref, str) or ref.count(":") != 1 or not all(ref.split(":")):
        raise ValueError(
            f"strategy {ref!r} must be an importable 'module:fn' string (for example "
            "'my_strategies:sma_trend')"
        )
    mod_name, attr = ref.split(":")
    try:
        obj: Any = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ValueError(f"cannot import strategy {ref!r}: {exc}") from exc
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(
                f"cannot import strategy {ref!r}: module {mod_name!r} has no {attr!r}"
            ) from exc
    if not callable(obj):
        raise ValueError(f"strategy {ref!r} is not callable")
    return obj


def _importable_ref(fn: Callable) -> str | None:
    """``module:qualname`` when ``fn`` is a top-level function a spawned worker can import."""
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    if not mod or not qual or mod == "__main__" or "<" in qual:
        return None
    try:
        pickle.dumps(fn)
        return f"{mod}:{qual}" if _import_strategy(f"{mod}:{qual}") is fn else None
    except Exception:
        return None


def _resolve_strategy(strategy, workers: int) -> tuple[Callable, str | None, str]:
    """Return ``(callable, importable_ref_or_None, display_name)``."""
    if isinstance(strategy, str):
        return _import_strategy(strategy), strategy, strategy
    if not callable(strategy):
        raise ValueError("strategy must be a callable or an importable 'module:fn' string")
    ref = _importable_ref(strategy)
    if workers > 1 and ref is None:
        raise ValueError(
            "with workers > 1 the strategy must be importable by a spawned worker: pass a "
            "'module:fn' string (or a top-level function of an importable module, not a "
            "lambda, a nested function or a function defined in __main__)"
        )
    return strategy, ref, ref or getattr(strategy, "__name__", "custom")


def default_workers(n_paths: int, cpu_count: int | None = None) -> int:
    """``min(cpu_count - 1, n_paths)``, at least 1 (1 means in-process)."""
    cpus = os.cpu_count() or 1 if cpu_count is None else cpu_count
    return max(1, min(cpus - 1, int(n_paths)))


# --- worker side -------------------------------------------------------------------------------

_STATE: dict = {}


def _init_worker(payload: dict) -> None:
    with contextlib.suppress(ValueError, OSError):  # Ctrl-C is handled by the parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    ref = payload.get("strategy_ref")
    _STATE.clear()
    _STATE.update(payload)
    _STATE["strategy"] = _import_strategy(ref) if ref else payload["strategy"]


def _eq_grid(equity: pd.Series) -> np.ndarray:
    e = equity.to_numpy(float)
    pos = np.round(np.linspace(0, len(e) - 1, EQ_POINTS)).astype(int)
    return e[pos] / e[0]


def _diagnostics(path) -> dict:
    crash = np.asarray(path.phase) == 1
    out = {}
    for a, col in path.returns.items():
        v = col.to_numpy(float)
        out[f"{CRASH_PREFIX}{a}"] = float(np.expm1(v[crash].sum()))
        out[f"{HORIZON_PREFIX}{a}"] = float(np.expm1(v.sum()))
    for f, col in path.factors.items():
        out[f"{FACTOR_PREFIX}{f}"] = float(col.to_numpy(float)[crash].sum())
    return out


def _compute(state: dict, i: int) -> tuple[int, dict, dict, np.ndarray]:
    path = generate_path(state["cal"], state["spec"], i, state["seed"], state["clock"],
                         dial=state["dial"])
    row, equity = run_on_path(
        state["strategy"], state["history"], path, i, state["clock"], mode=state["mode"],
        benchmark=state["benchmark"], ruin_level=state["ruin_level"], **state["run_kwargs"],
    )
    return i, row, _diagnostics(path), _eq_grid(equity)


def _work(i: int) -> tuple[int, dict, dict, np.ndarray]:
    return _compute(_STATE, i)


# --- files -------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def _line(values: Iterable) -> bytes:
    return (",".join(values) + "\n").encode("utf-8")


class _Appender:
    """O_APPEND writer: one complete line per ``os.write``, fsynced."""

    def __init__(self, path: Path) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        self.fd = os.open(path, flags, 0o644)

    def write(self, data: bytes) -> None:
        os.write(self.fd, data)
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def _recover(path: Path) -> tuple[list[str] | None, set[int]]:
    """Header and finished path indices of ``path``; discards an unterminated last line."""
    if not path.exists():
        return None, set()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1  # 0 when not even the header was completed
        os.truncate(path, keep)
        data = data[:keep]
    if not data:
        return None, set()
    lines = data.decode("utf-8").split("\n")[:-1]
    header = lines[0].split(",")
    if "path" not in header:
        raise ValueError(f"{path}: not a crisispaths results file (no 'path' column)")
    col = header.index("path")
    done: set[int] = set()
    for n, ln in enumerate(lines[1:], start=2):
        f = ln.split(",")
        if len(f) != len(header):
            raise ValueError(f"{path}: line {n} has {len(f)} fields, expected {len(header)}")
        done.add(int(f[col]))
    return header, done


def _read_table(path: Path, n_paths: int | None = None, keep_eq: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, float_precision="round_trip")
    df = df.drop_duplicates("path").sort_values("path")
    if n_paths is not None:
        df = df[df["path"] < n_paths]
    if not keep_eq:
        df = df.drop(columns=[c for c in df.columns if c.startswith(EXTRA_PREFIXES)])
    return df.reset_index(drop=True)


def load_results(out_dir, key: str, n_paths: int | None = None, *, keep_eq: bool = False):
    """Rows of ``results_<key>.csv`` sorted by path (``eq_*`` columns dropped by default)."""
    return _read_table(Path(out_dir) / f"results_{key}.csv", n_paths, keep_eq)


def _nested(df: pd.DataFrame) -> list:
    a = df.to_numpy(float)
    return [[float(x) if np.isfinite(x) else None for x in r] for r in a]


def _loadings_json(cal: Calibration) -> dict | None:
    L = cal.loadings
    if L is None:
        return None
    regimes = list(L.betas)
    first = L.betas[regimes[0]]
    return {
        "regimes": regimes,
        "assets": [str(a) for a in first.index],
        "factors": [str(f) for f in first.columns],
        "betas": {r: _nested(L.betas[r]) for r in regimes},
        "std_errors": {r: _nested(L.std_errors[r]) for r in regimes},
        "n_obs": {r: [int(x) for x in L.n_obs[r]] for r in regimes},
        "shrinkage": {r: [None if not np.isfinite(x) else float(x) for x in L.shrinkage[r]]
                      for r in regimes},
        "unaffected": sorted(L.unaffected),
        "imputed": sorted(L.imputed),
        "warnings": list(L.warnings),
        "imputation": {a: dict(v) for a, v in L.metadata.get("imputation", {}).items()},
    }


# --- one scenario ------------------------------------------------------------------------------


def _scenario(
    prices, strategy, spec: ScenarioSpec, *, n_paths, seed, clock, out_dir: Path, workers,
    resume, calibration, factors, source, dial, mode, benchmark, ruin_level, check_lookahead,
    on_lookahead, progress, run_kwargs, calm_drift=None,
) -> pd.DataFrame:
    from . import __version__

    fn, ref, name = _resolve_strategy(strategy, workers)
    clock, cal, cal_hash, history = prepare_run(
        prices, spec, clock=clock, calibration=calibration, factors=factors, source=source,
        calm_drift=calm_drift,
    )
    key = spec.key
    csv_path = out_dir / f"results_{key}.csv"
    pin_path = out_dir / f"pin_{key}.json"
    pin = {
        "spec_content_hash": spec.content_hash,
        "calibration_hash": cal_hash,
        "seed": int(seed),
        "package_version": __version__,
        "dial": float(dial),
        "mode": mode,
        "ruin_level": float(ruin_level),
        "benchmark": None if benchmark is None else getattr(benchmark, "__name__", "custom"),
        "run_kwargs": {k: (v if isinstance(v, (int, float, str, type(None))) else repr(v))
                       for k, v in sorted(run_kwargs.items())},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for p in (csv_path, pin_path):
            p.unlink(missing_ok=True)
    if pin_path.exists():
        old = json.loads(pin_path.read_text(encoding="utf-8"))
        diff = [k for k in sorted(set(old) | set(pin)) if old.get(k) != pin.get(k)]
        if diff:
            detail = "; ".join(f"{k}: pinned {old.get(k)!r}, now {pin.get(k)!r}" for k in diff)
            raise PinMismatchError(
                f"results for {key!r} in {out_dir} come from a different run ({detail}). "
                "Use another out_dir or resume=False (--no-resume) to start over."
            )
    header, done = _recover(csv_path)
    if done and not pin_path.exists():
        raise PinMismatchError(
            f"{csv_path} has results but no pin file {pin_path.name}; refusing to mix runs. "
            "Use resume=False (--no-resume) to start over."
        )
    if not done:
        header = None
        csv_path.unlink(missing_ok=True)
    _atomic_write_json(pin_path, pin)
    _atomic_write_json(out_dir / f"manifest_{key}.json", {
        **pin,
        "key": key,
        "family": spec.metadata.get("family", spec.name),
        "axis_choices": spec.metadata.get("axis_choices", {}),
        "overrides": spec.metadata.get("overrides", {}),
        "merged_choices": spec.metadata.get("merged_choices", []),
        "skipped": spec.metadata.get("skipped", []),
        "spec": _to_dict(spec),
        "strategy": name,
        "n_paths": int(n_paths),
        "costs_bps": float(run_kwargs.get("costs_bps", 0.0)),
        "cost_multiplier": float(run_kwargs.get("cost_multiplier", 1.0)),
        "bars_per_year": float(clock.bars_per_year),
        "bars_per_day": float(clock.bars_per_day),
        "calibration_bars_per_year": float(cal.bars_per_year),
        "intraday_profile": None if cal.intraday is None else cal.intraday.source,
        "overnight_gap": bool(clock.overnight_gap),
        "analog_exact": uses_exact_analog(spec),
        "calibration_source": cal.source,
        "calm_drift": cal.calm_drift,
        "calibration_warnings": list(cal.warnings),
        "calibration_stress_windows": [list(w) for w in cal.stress_windows],
        "loadings": _loadings_json(cal),
        "assets": list(cal.assets),
        "factors": list(cal.factors),
    })

    if check_lookahead and mode == "weights":
        rep = _audit(fn, history)
        if not rep.passed:
            if on_lookahead == "raise":
                raise LookaheadError(rep.message())
            import warnings

            warnings.warn(rep.message(), UserWarning, stacklevel=3)

    todo = [i for i in range(n_paths) if i not in done]
    finished = sum(1 for i in done if i < n_paths)
    state = {
        "strategy": fn, "strategy_ref": ref, "history": history, "cal": cal, "spec": spec,
        "seed": int(seed), "clock": clock, "dial": dial, "mode": mode, "benchmark": benchmark,
        "ruin_level": ruin_level, "run_kwargs": dict(run_kwargs),
    }
    if todo:
        app = _Appender(csv_path)
        try:
            n_workers = min(workers, len(todo))
            if n_workers <= 1:
                stream: Iterable = (_compute(state, i) for i in todo)
                pool = None
            else:
                pool = ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=mp.get_context("spawn"),
                    initializer=_init_worker, initargs=(state,),
                )
                stream = pool.map(_work, todo)
            try:
                for _, row, diag, grid in stream:
                    cols = [*row, *diag, *(f"{EQ_PREFIX}{k:02d}" for k in range(EQ_POINTS))]
                    if header is None:
                        header = cols
                        app.write(_line(header))
                    elif cols != header:
                        raise ValueError(
                            f"{csv_path.name}: columns of the new rows differ from the file "
                            "header (a different benchmark or version?)"
                        )
                    app.write(_line([*map(_fmt, row.values()), *map(_fmt, diag.values()),
                                     *map(_fmt, grid)]))
                    finished += 1
                    if progress is not None:
                        progress(key, finished, n_paths)
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
        finally:
            app.close()
    table = _read_table(csv_path, n_paths) if csv_path.exists() else pd.DataFrame()
    if len(table):
        table.insert(0, "scenario", key)
    return table


# --- public API --------------------------------------------------------------------------------


def run_mc(
    prices: pd.DataFrame,
    strategy,
    specs,
    *,
    n_paths: int = 1000,
    seed: int = 0,
    clock: Clock | None = None,
    out_dir,
    workers: int | None = None,
    costs_bps: float = 0.0,
    cost_multiplier: float = 1.0,
    resume: bool = True,
    calibration: Calibration | None = None,
    factors: pd.DataFrame | None = None,
    source=None,
    calm_drift: str | None = None,
    dial: float = 1.0,
    mode: str = "weights",
    benchmark: Callable | None = buy_and_hold_equal_weight,
    ruin_level: float = 0.5,
    check_lookahead: bool = True,
    on_lookahead: str = "raise",
    progress: Callable[[str, int, int], None] | None = None,
    **run_kwargs,
) -> pd.DataFrame:
    """Run ``strategy`` on ``n_paths`` paths of each scenario, resumably and in parallel.

    ``specs`` is a scenario (spec, alias or family name) or a list of them.
    ``strategy`` is an importable ``"module:fn"`` string, required when
    ``workers > 1`` (a top-level function of an importable module is converted
    to one). ``workers=None`` uses ``min(cpu_count - 1, n_paths)``;
    ``workers=1`` runs in this process. ``progress(key, done, total)`` is called
    after every path. On ``KeyboardInterrupt`` the finished rows stay on disk
    and the same call resumes them.

    Returns the combined table: a ``scenario`` column plus the columns of
    ``backtest(...).table``, rows sorted by scenario order then path. The
    result keeps ``attrs["out_dir"]`` for :func:`crisispaths.report.write_report`.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if workers is not None and workers < 1:
        raise ValueError("workers must be >= 1 or None")
    items: Sequence = list(specs) if isinstance(specs, (list, tuple)) else [specs]
    if not items:
        raise ValueError("no scenarios given")
    resolved = [resolve_spec(s) for s in items]
    keys = [s.key for s in resolved]
    if len(set(keys)) != len(keys):
        raise ValueError(f"scenario keys must be unique, got {keys}")
    n_workers = default_workers(n_paths) if workers is None else int(workers)
    out = Path(out_dir)
    kw = dict(run_kwargs, costs_bps=float(costs_bps), cost_multiplier=float(cost_multiplier))
    frames = []
    first = True
    for spec in resolved:
        frames.append(_scenario(
            prices, strategy, spec, n_paths=n_paths, seed=seed, clock=clock, out_dir=out,
            workers=n_workers, resume=resume, calibration=calibration, factors=factors,
            source=source, dial=dial, mode=mode, benchmark=benchmark, ruin_level=ruin_level,
            check_lookahead=check_lookahead and first, on_lookahead=on_lookahead,
            progress=progress, run_kwargs=kw, calm_drift=calm_drift,
        ))
        first = False
    table = pd.concat(frames, ignore_index=True)
    table.attrs["out_dir"] = str(out)
    return table


def run_sweep(
    prices: pd.DataFrame,
    strategy,
    family,
    axes: dict,
    *,
    out_dir,
    max_scenarios: int = DEFAULT_MAX_SCENARIOS,
    overrides: dict | None = None,
    **kw,
) -> pd.DataFrame:
    """Run every distinct scenario of a grid and return one long table.

    ``axes`` maps axis name to an option or a list of options (lists are swept,
    scalars fixed, omitted axes take their default). Combinations that compose
    to identical content run once (``enumerate_scenarios(dedupe=True)``). The
    table has the columns of :func:`run_mc` plus one column per axis (the
    choices), ``merged_choices`` (JSON list of the combinations merged into the
    scenario) and ``content_hash``. ``max_scenarios`` guards against silently
    huge sweeps; nothing is written when it is exceeded.
    """
    if max_scenarios < 1:
        raise ValueError("max_scenarios must be >= 1")
    rep = grid_report(family, overrides=overrides, **axes)
    if rep.n_unique > max_scenarios:
        raise ValueError(
            f"the sweep has {rep.n_total} combinations and {rep.n_unique} distinct scenarios, "
            f"more than max_scenarios={max_scenarios}. Each scenario runs n_paths paths, so "
            "this is expensive: narrow the axes, or pass a larger max_scenarios to confirm."
        )
    pairs = list(enumerate_scenarios(family, overrides=overrides, dedupe=True, **axes))
    out = Path(out_dir)
    parts = []
    for n, (choices, spec) in enumerate(pairs):
        sub = dict(kw)
        if n:
            sub["check_lookahead"] = False
        t = run_mc(prices, strategy, spec, out_dir=out, **sub)
        if t.empty:
            continue
        clash = [a for a in choices if a in t.columns]
        if clash:
            raise ValueError(f"axis name(s) {clash} collide with result columns")
        merged = json.dumps(spec.metadata.get("merged_choices", []), sort_keys=True)
        for a, v in choices.items():
            t[a] = v
        t["merged_choices"] = merged
        t["content_hash"] = spec.content_hash
        parts.append(t)
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.attrs["out_dir"] = str(out)
    table.attrs["grid"] = {"n_total": rep.n_total, "n_unique": rep.n_unique}
    return table


__all__ = [
    "DEFAULT_MAX_SCENARIOS",
    "PinMismatchError",
    "default_workers",
    "load_results",
    "run_mc",
    "run_sweep",
]


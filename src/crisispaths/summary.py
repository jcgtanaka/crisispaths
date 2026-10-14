"""Machine-readable summary (``summary.json``, ``schema_version`` 1).

The document mirrors REPORT.md: banner, caveats, per-scenario provenance,
judgment parameters, loadings, path sanity, metrics with intervals and the
benchmark comparison, and for sweeps the per-scenario variant table with
per-axis-option ranges. It NEVER contains an aggregate across scenarios (no
mean, pooled or expected figure): scenarios are conditional what-ifs.

The JSON Schema is shipped as ``crisispaths/schema/summary.schema.json``.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import CI_LEVEL, summarize
from .report import (
    BANNER,
    EXTRA_PREFIXES,
    _caveats,
    _load,
    benchmark_stats,
    path_sanity,
)

SCHEMA_VERSION = 1


def load_schema() -> dict:
    """The packaged JSON Schema of ``summary.json``."""
    text = (resources.files("crisispaths") / "schema" / "summary.schema.json").read_text("utf-8")
    return json.loads(text)


def _num(x):
    """JSON-safe float: NaN and inf become ``None``."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _num(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def judgment_parameters(man: dict) -> list[dict]:
    """Every ``calibrated: false`` parameter of the spec, with its rationale."""
    spec = man["spec"]
    out = []
    for name, a in (spec.get("anchors") or {}).items():
        if a.get("calibrated") is False:
            out.append({"path": f"anchors.{name}", "cum_dist": a.get("cum_dist"),
                        "params": a.get("params", {}), "rationale": a.get("rationale", "")})
    aft = spec.get("aftermath") or {}
    if aft.get("calibrated") is False:
        keep = ("mode", "bear_annual_drift", "recovery_fraction", "recovery_period")
        out.append({"path": "aftermath", "cum_dist": None,
                    "params": {k: aft[k] for k in keep if k in aft},
                    "rationale": aft.get("rationale", "")})
    for name, m in (spec.get("param_meta") or {}).items():
        if m.get("calibrated") is False:
            out.append({"path": f"param_meta.{name}", "cum_dist": None, "params": {},
                        "rationale": m.get("rationale", "")})
    return out


def _loadings(man: dict) -> dict | None:
    L = man.get("loadings")
    if not L:
        return None
    un, imp = set(L["unaffected"]), set(L["imputed"])
    imputation = L.get("imputation") or {}
    rows = []
    for i, asset in enumerate(L["assets"]):
        rows.append({
            "asset": asset,
            "n_obs": {r: int(L["n_obs"][r][i]) for r in L["regimes"]},
            "shrinkage": {r: _num(L["shrinkage"][r][i]) for r in L["regimes"]},
            "betas": {r: {f: _num(L["betas"][r][i][j]) for j, f in enumerate(L["factors"])}
                      for r in L["regimes"]},
            "std_errors": {r: {f: _num(L["std_errors"][r][i][j])
                               for j, f in enumerate(L["factors"])} for r in L["regimes"]},
            "unaffected": asset in un,
            "imputed": asset in imp,
            "amplification": _num((imputation.get(asset) or {}).get("amplification")),
        })
    return {"regimes": list(L["regimes"]), "factors": list(L["factors"]), "assets": rows,
            "warnings": list(L.get("warnings", []))}


def _metrics(man: dict, tab: pd.DataFrame, s, n_boot: int) -> dict:
    q = s.quantiles
    rec = s.recovery
    bstats = benchmark_stats(man, tab, n_boot)
    return {
        "median_terminal_wealth": _num(tab["terminal_wealth"].median()),
        "quantiles": [{"metric": r.metric, "quantile": float(r.quantile), "value": _num(r.value),
                       "ci_low": _num(r.ci_low), "ci_high": _num(r.ci_high)}
                      for r in q.itertuples()],
        "cvar": {m: {"tail": c["tail"], "alpha": float(c["alpha"]), "value": _num(c["value"]),
                     "ci_low": _num(c["ci_low"]), "ci_high": _num(c["ci_high"])}
                 for m, c in s.cvar.items()},
        "ruin": {"p": _num(s.ruin["p"]), "n_ruined": int(s.ruin["n_ruined"]),
                 "n": int(s.ruin["n"]), "ci_low": _num(s.ruin["ci_low"]),
                 "ci_high": _num(s.ruin["ci_high"])},
        "recovery": {
            "share_recovered": _num(rec["share_recovered"]),
            "share_recovered_ci": [_num(v) for v in rec["share_recovered_ci"]],
            "median_time_bars": _num(rec["median_time"]),
            "median_time_ci": [_num(v) for v in rec["median_time_ci"]],
            "horizon_bars": _num(rec["horizon_bars"]),
        },
        "censoring": {"n_censored": int(rec["n_censored"]), "n": int(rec["n"])},
        "benchmark_comparison": None if bstats is None else [
            {"statistic": r["label"], "strategy": _num(r["strategy"]),
             "benchmark": _num(r["benchmark"]), "difference": _num(r["difference"]),
             "ci_low": _num(r["ci_low"]), "ci_high": _num(r["ci_high"])} for r in bstats
        ],
        "ci_level": CI_LEVEL,
    }


def build_summary(result_or_dir, *, n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """The summary document (a plain dict) for a runner output."""
    from . import __version__

    src, scen = _load(result_or_dir)
    summaries = {}
    for man, tab in scen:
        if tab.empty:
            continue
        core = tab.drop(columns=[c for c in tab.columns if c.startswith(EXTRA_PREFIXES)])
        summaries[man["key"]] = summarize(core, alpha=alpha, n_boot=n_boot, seed=int(man["seed"]),
                                          analog_exact=bool(man.get("analog_exact")))
    caveats = [ln.strip().lstrip("- ").strip() for ln in _caveats(scen, summaries).splitlines()
               if ln.strip()]
    scenarios = []
    for man, tab in scen:
        if tab.empty:
            continue
        bpd = float(man.get("bars_per_day", 1.0))
        bpy = float(man["bars_per_year"])
        scenarios.append({
            "key": man["key"],
            "family": man.get("family", ""),
            "axis_choices": man.get("axis_choices") or {},
            "overrides": man.get("overrides") or {},
            "merged_choices": man.get("merged_choices") or [],
            "skipped_options": man.get("skipped") or [],
            "content_hash": man["spec_content_hash"],
            "calibration_hash": man["calibration_hash"],
            "seed": int(man["seed"]),
            "package_version": man["package_version"],
            "calm_drift": man.get("calm_drift", "historical"),
            "label": man["spec"].get("label", ""),
            "frequency": {"bars_per_year": bpy, "bars_per_day": bpd,
                          "kind": "intraday" if bpd > 1.0 else "daily_or_coarser"},
            "profile_source": man.get("intraday_profile"),
            "n_paths_requested": int(man.get("n_paths", len(tab))),
            "n_paths": int(len(tab)),
            "costs_bps": float(man["costs_bps"]),
            "cost_multiplier": float(man["cost_multiplier"]),
            "benchmark": man.get("benchmark"),
            "judgment_parameters": judgment_parameters(man),
            "loadings": _loadings(man),
            "path_sanity": [
                {"asset": r["asset"], "crash_leg_median": _num(r["crash_median"]),
                 "horizon_median": _num(r["horizon_median"]),
                 "implied_by_stress_loadings": _num(r["expected"]),
                 "imputed": bool(r["imputed"]), "amplification": _num(r["amplification"]),
                 "warning": r["warning"] or None}
                for r in path_sanity(man, tab)
            ],
            "metrics": _metrics(man, tab, summaries[man["key"]], n_boot),
        })
    variants = None
    live = [s for s in scenarios if s["axis_choices"]]
    if len(live) >= 2:
        rows = []
        for s in live:
            t = next(t for m, t in scen if m["key"] == s["key"])
            ex = (float(t["excess_terminal_wealth"].median())
                  if "excess_terminal_wealth" in t.columns else None)
            rows.append({
                "key": s["key"], "axis_choices": s["axis_choices"],
                "median_terminal_wealth": s["metrics"]["median_terminal_wealth"],
                "p_ruin": s["metrics"]["ruin"]["p"],
                "median_max_drawdown": _num(t["max_drawdown_from_peak"].median()),
                "median_excess_wealth": _num(ex),
            })
        ranges = []
        axes = list(dict.fromkeys(a for r in rows for a in r["axis_choices"]))
        for a in axes:
            opts = list(dict.fromkeys(r["axis_choices"].get(a) for r in rows))
            if len(opts) < 2:
                continue
            for o in opts:
                sel = [r for r in rows if r["axis_choices"].get(a) == o]
                tw = [r["median_terminal_wealth"] for r in sel]
                dd = [r["median_max_drawdown"] for r in sel]
                ranges.append({"axis": a, "option": str(o), "n_scenarios": len(sel),
                               "median_terminal_wealth_min": min(tw),
                               "median_terminal_wealth_max": max(tw),
                               "median_max_drawdown_min": min(dd),
                               "median_max_drawdown_max": max(dd)})
        variants = {"scenarios": rows, "axis_ranges": ranges}
    doc = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "banner": BANNER,
        "source_directory": src.name,
        "caveats": caveats,
        "scenarios": scenarios,
        "variants": variants,
    }
    return _clean(doc)


def write_summary(result_or_dir, out_dir, *, n_boot: int = 1000, alpha: float = 0.05) -> Path:
    """Write ``summary.json`` into ``out_dir`` and return its path."""
    doc = build_summary(result_or_dir, n_boot=n_boot, alpha=alpha)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


__all__ = ["SCHEMA_VERSION", "build_summary", "judgment_parameters", "load_schema",
           "write_summary"]

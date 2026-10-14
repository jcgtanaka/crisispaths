"""REPORT.md writer.

Section order (fixed, tested): banner, caveats, scenario provenance, factor
loadings, results, and for a sweep a variant comparison. The report never
reports an expected return across scenarios: scenarios are conditional
what-ifs and no probability is attached to any of them. The variant comparison
shows each scenario's own numbers and, per axis option, the minimum and maximum
of those numbers (how much the answer moves with a judgment call), never a
pooled or averaged figure.

Plots (matplotlib, optional ``[report]`` extra) are written next to the report;
without matplotlib the report says so and stays complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import CI_LEVEL, MIN_PATHS_FOR_CIS, summarize
from .runner import (
    CRASH_PREFIX,
    EQ_POINTS,
    EQ_PREFIX,
    EXTRA_PREFIXES,
    FACTOR_PREFIX,
    HORIZON_PREFIX,
    load_results,
)

BANNER = (
    "Conditional on this scenario occurring. These are what-ifs, not forecasts. "
    "No probability is attached to any scenario."
)
HIGH_TURNOVER_PER_YEAR = 12.0  # sum of |weight change| per year, in multiples of capital
DAILY_BARS_PER_YEAR = 252.0
_MD_ESCAPE = str.maketrans({"|": "\\|"})


# --- helpers -----------------------------------------------------------------------------------


def _pyplot():
    """matplotlib.pyplot with the Agg backend, or ``None`` when it is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def intraday_warning(bars_per_year: float, annual_turnover: float) -> str | None:
    """Warning text when bars are finer than daily and the strategy trades a lot."""
    if bars_per_year > DAILY_BARS_PER_YEAR and annual_turnover > HIGH_TURNOVER_PER_YEAR:
        return (
            f"intraday optimism: {bars_per_year:g} bars per year with high turnover "
            f"(about {annual_turnover:.0f}x capital per year). Paths are simulated at the "
            "calibration frequency from daily-style dynamics; spreads, slippage, queue position "
            "and gaps at this trading speed are not modelled, so these results are optimistic"
        )
    return None


INTRADAY_CAVEATS = (
    "  - intraday limits: within a session the path is a bridge between simulated daily "
    "closes, so there are no jumps inside the session, no trading halts or liquidity gaps, the "
    "cost model is flat per unit of turnover, the Epps effect is not modelled (within-day "
    "cross-asset correlation is the daily one), and strategies that react to intraday moves "
    "(stop-losses, intraday breakouts) look better than they would in reality."
)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    def esc(x) -> str:
        return str(x).translate(_MD_ESCAPE)

    lines = ["| " + " | ".join(esc(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _num(x, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _pct(x, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def _with_ci(v, lo, hi, fmt=_num) -> str:
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"


def _choices(man: dict) -> str:
    c = man.get("axis_choices") or {}
    return ", ".join(f"{a}={o}" for a, o in c.items()) if c else "(alias or custom spec)"


def _paired_ci(a, b, stat, n_boot: int, seed: int, level: float = CI_LEVEL):
    """Percentile bootstrap of ``stat(a*) - stat(b*)`` resampling paths jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    reps = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(250, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        reps[done:done + m] = stat(a[idx]) - stat(b[idx])
        done += m
    lo, hi = np.percentile(reps, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def _med(x):
    return np.median(x, axis=1)


def _mean(x):
    return np.mean(x, axis=1)


# --- loading -----------------------------------------------------------------------------------


def _load(source) -> tuple[Path, list[tuple[dict, pd.DataFrame]]]:
    order: list[str] | None = None
    if isinstance(source, pd.DataFrame):
        out = source.attrs.get("out_dir")
        if out is None:
            raise ValueError(
                "this frame has no out_dir; pass the table returned by run_mc/run_sweep or the "
                "output directory itself"
            )
        if "scenario" in source.columns:
            order = list(dict.fromkeys(source["scenario"]))
        src = Path(out)
    else:
        src = Path(source)
    manifests = {}
    for f in sorted(src.glob("manifest_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        manifests[m["key"]] = m
    if not manifests:
        raise ValueError(f"no results found in {src} (no manifest_*.json)")
    keys = [k for k in (order or sorted(manifests)) if k in manifests]
    keys += [k for k in sorted(manifests) if k not in keys and order is None]
    out: list[tuple[dict, pd.DataFrame]] = []
    for k in keys:
        man = manifests[k]
        f = src / f"results_{k}.csv"
        tab = load_results(src, k, man.get("n_paths"), keep_eq=True) if f.exists() else None
        if tab is None or tab.empty:
            tab = pd.DataFrame()
        out.append((man, tab))
    if all(t.empty for _, t in out):
        raise ValueError(f"no results found in {src} (the results files are empty)")
    return src, out


# --- sections ----------------------------------------------------------------------------------


def _judgment_items(man: dict) -> list[str]:
    spec = man["spec"]
    items = []
    for name, a in (spec.get("anchors") or {}).items():
        if a.get("calibrated") is False:
            items.append(
                f"- JUDGMENT `anchors.{name}`: {a['cum_dist']} "
                f"{json.dumps(a.get('params', {}), sort_keys=True)}"
                + (f" -- {a['rationale']}" if a.get("rationale") else "")
            )
    aft = spec.get("aftermath") or {}
    if aft.get("calibrated") is False:
        keys = ("mode", "bear_annual_drift", "recovery_fraction", "recovery_period")
        shown = {k: aft[k] for k in keys if k in aft}
        items.append(
            f"- JUDGMENT `aftermath`: {json.dumps(shown, sort_keys=True)}"
            + (f" -- {aft['rationale']}" if aft.get("rationale") else "")
        )
    for name, m in (spec.get("param_meta") or {}).items():
        if m.get("calibrated") is False:
            items.append(
                f"- JUDGMENT `param_meta.{name}`"
                + (f" -- {m['rationale']}" if m.get("rationale") else "")
            )
    return items


def _caveats(scen: list[tuple[dict, pd.DataFrame]], summaries: dict) -> str:
    lines = [
        "- Scenarios are conditional what-ifs. Nothing here says a scenario will happen or "
        "how likely it is.",
    ]
    costs = {(m["costs_bps"], m["cost_multiplier"]) for m, _ in scen}
    for c, mult in sorted(costs):
        note = (
            "zero costs: crisis costs are not zero, results are optimistic"
            if c == 0 or mult == 0 else
            f"effective {c * mult:g} bps per unit of turnover"
        )
        lines.append(f"- Costs: costs_bps={c}, cost_multiplier={mult}; {note}.")
    if not any(float(m.get("bars_per_day", 1.0)) > 1.0 for m, _ in scen):
        lines.append(
            "- Paths are simulated at the calibration bar frequency (daily for daily prices)."
        )
    for man, _ in scen:
        bpd = float(man.get("bars_per_day", 1.0))
        bpy = float(man["bars_per_year"])
        if bpd > 1.0:
            src = man.get("intraday_profile") or "flat"
            lines.append(
                f"- `{man['key']}`: frequency {bpd:g} bars per session, {bpy:g} bars per year. "
                "Paths are simulated daily and disaggregated into bars by a Brownian bridge "
                f"(intraday profile: {src}"
                + ("; NO intraday history was supplied, every bar has the same variance share"
                   if src == "flat" else "") + ")."
            )
            lines.append(INTRADAY_CAVEATS)
        else:
            lines.append(f"- `{man['key']}`: frequency {bpy:g} bars per year (daily or coarser).")
    for man, tab in scen:
        key = man["key"]
        n = len(tab)
        req = int(man.get("n_paths", n))
        lines.append(f"- `{key}`: {n} paths simulated.")
        if n < req:
            lines.append(
                f"  - PARTIAL: only {n} of {req} requested paths are on disk; rerun with "
                "resume to finish them."
            )
        if n == 0:
            continue
        s = summaries[key]
        if n < MIN_PATHS_FOR_CIS:
            lines.append(
                f"  - only {n} paths (< {MIN_PATHS_FOR_CIS}): quantile, CVaR and recovery "
                "intervals are unstable; use at least 200 paths for tail statistics."
            )
        if man.get("analog_exact"):
            lines.append(
                "  - analog exact mode: the crash replays one historical window in every path, "
                "so the intervals are conditional on that single history and understate "
                "scenario uncertainty."
            )
        nc = s.recovery["n_censored"]
        lines.append(
            f"  - censoring: {nc} of {n} paths did not recover within the horizon (censored); "
            "recovery is estimated with Kaplan-Meier, never a mean of observed times."
            if nc else
            f"  - censoring: all {n} paths recovered within the horizon."
        )
        if {"total_turnover", "horizon_bars"} <= set(tab.columns):
            years = float(tab["horizon_bars"].iloc[0]) / float(man["bars_per_year"])
            annual = float(tab["total_turnover"].mean()) / years if years > 0 else 0.0
            w = intraday_warning(float(man["bars_per_year"]), annual)
            if w:
                lines.append(f"  - {w}.")
    return "\n".join(lines)


def _provenance(scen: list[tuple[dict, pd.DataFrame]]) -> str:
    rows = []
    for man, _ in scen:
        rows.append([
            man["key"], man.get("family", ""), _choices(man),
            json.dumps(man.get("overrides") or {}, sort_keys=True),
            man["spec_content_hash"][:12], str(man["seed"]), man["package_version"],
            man["calibration_hash"][:12], man.get("calm_drift", "historical (legacy run)"),
            man["spec"].get("label", ""),
        ])
    parts = [_md_table(
        ["scenario", "family", "axis choices", "overrides", "content hash", "seed", "version",
         "calibration hash", "calm drift", "label"], rows)]
    parts.append(
        "Calm drift: `zero` means the calm-period returns used after a V recovery and before "
        "the onset have their historical mean removed (volatility, correlation and tails kept); "
        "`historical` keeps each asset's calm-period trend."
    )
    parts.append(
        "Parameters marked `calibrated: false` are judgment calls with a stated rationale, "
        "not estimates from data."
    )
    per = {m["key"]: _judgment_items(m) for m, _ in scen}
    common = [x for x in next(iter(per.values()), []) if all(x in v for v in per.values())]
    if len(scen) > 1 and common:
        parts.append("**Judgment parameters shared by every scenario**\n\n" + "\n".join(common))
    else:
        common = []
    for man, _ in scen:
        items = [x for x in per[man["key"]] if x not in common]
        extra = []
        if man.get("skipped"):
            extra.append(f"- options skipped because their requirements were not met: "
                         f"{', '.join(man['skipped'])}")
        if man.get("merged_choices"):
            extra.append(f"- identical content, merged: {json.dumps(man['merged_choices'])}")
        body = "\n".join(items + extra) if items or extra else "- (no further judgment flags)"
        parts.append(f"**{man['key']}**\n\n{body}")
    return "\n\n".join(parts)


def _loadings(scen: list[tuple[dict, pd.DataFrame]]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for man, _ in scen:
        h = man["calibration_hash"]
        L = man.get("loadings")
        if h in seen:
            continue
        seen.add(h)
        scen_for = [m["key"] for m, _ in scen if m["calibration_hash"] == h]
        parts.append(f"Calibration `{h[:12]}` (scenarios: {', '.join(scen_for)}).")
        if L is None:
            parts.append("Loadings are not available for this calibration.")
            continue
        un, imp = set(L["unaffected"]), set(L["imputed"])
        for regime in L["regimes"]:
            rows = []
            for ai, asset in enumerate(L["assets"]):
                cells = []
                for fi in range(len(L["factors"])):
                    b = L["betas"][regime][ai][fi]
                    se = L["std_errors"][regime][ai][fi]
                    cells.append("n/a" if b is None else
                                 f"{b:.2f} ({'n/a' if se is None else f'{se:.2f}'})")
                flag_list = []
                if asset in un:
                    flag_list.append("unaffected")
                if asset in imp:
                    amp = (L.get("imputation") or {}).get(asset, {}).get("amplification")
                    flag_list.append(
                        "IMPUTED (no data in stress windows; borrowed from neighbours, "
                        f"amplification s={'n/a' if amp is None else f'{amp:.2f}'})"
                    )
                flags = ", ".join(flag_list) or "-"
                sh = L["shrinkage"][regime][ai]
                rows.append([asset, str(L["n_obs"][regime][ai]), _num(sh, 2), flags, *cells])
            parts.append(
                f"Regime `{regime}` (beta and standard error per factor):\n\n"
                + _md_table(["asset", "n_obs", "shrinkage share", "unaffected / imputed",
                             *L["factors"]], rows)
            )
        if L["warnings"]:
            parts.append("Warnings:\n\n" + "\n".join(f"- {w}" for w in L["warnings"]))
        if man.get("calibration_warnings"):
            extra = [w for w in man["calibration_warnings"] if w not in set(L["warnings"])]
            if extra:
                parts.append("Calibration warnings:\n\n" + "\n".join(f"- {w}" for w in extra))
    return "\n\n".join(parts)


def _plots(plt, tab: pd.DataFrame, man: dict, summary, out: Path) -> list[tuple[str, str]]:
    key = man["key"]
    made: list[tuple[str, str]] = []
    grid_cols = [f"{EQ_PREFIX}{k:02d}" for k in range(EQ_POINTS)]
    horizon = int(tab["horizon_bars"].iloc[0])
    xs = np.round(np.linspace(0, horizon, EQ_POINTS)) / float(man["bars_per_year"])

    def save(fig, stem: str, title: str) -> None:
        name = f"{stem}_{key}.png"
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        made.append((title, name))

    if all(c in tab.columns for c in grid_cols):
        eq = tab[grid_cols].to_numpy(float)
        q = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(xs, q[0], q[4], alpha=0.2, label="5-95%")
        ax.fill_between(xs, q[1], q[3], alpha=0.35, label="25-75%")
        ax.plot(xs, q[2], lw=2, label="median")
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set(xlabel="years", ylabel="equity / start", title=f"Equity percentiles: {key}")
        ax.legend()
        save(fig, "fan", "Equity percentiles across paths")

        worst = tab.nsmallest(min(10, len(tab)), "terminal_wealth").index
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in worst:
            ax.plot(xs, tab.loc[i, grid_cols].to_numpy(float), lw=1)
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set(xlabel="years", ylabel="equity / start", title=f"Worst 10 paths: {key}")
        save(fig, "worst10", "Ten worst paths by terminal wealth")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tab["max_drawdown_from_peak"], bins=20, color="C3", alpha=0.8)
    ax.set(xlabel="max drawdown from peak", ylabel="paths", title=f"Drawdowns: {key}")
    save(fig, "drawdown", "Maximum drawdown from peak")

    km = summary.recovery["km"]
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.concatenate([[0.0], km["time"].to_numpy(float)])
    s = np.concatenate([[1.0], km["survival"].to_numpy(float)])
    ax.step(t, s, where="post")
    ax.set(xlabel="bars after the trough", ylabel="share not yet recovered",
           ylim=(0, 1.02), title=f"Recovery (Kaplan-Meier): {key}")
    save(fig, "recovery", "Time to recover, Kaplan-Meier")
    return made


def benchmark_stats(man: dict, tab: pd.DataFrame, n_boot: int) -> list[dict] | None:
    """Strategy against benchmark on the same paths (paired bootstrap), or ``None``."""
    if "bench_terminal_wealth" not in tab.columns:
        return None
    seed = int(man["seed"])
    specs = [
        ("median terminal wealth", tab["terminal_wealth"], tab["bench_terminal_wealth"],
         _med, _num),
        ("median max drawdown from peak", tab["max_drawdown_from_peak"],
         tab["bench_max_drawdown_from_peak"], _med, _pct),
        ("P(ruin)", tab["ruined"].astype(float), tab["bench_ruined"].astype(float), _mean, _pct),
    ]
    out = []
    for label, a, b, st, fmt in specs:
        lo, hi = _paired_ci(a, b, st, n_boot, seed)
        va = float(st(a.to_numpy(float)[None, :])[0])
        vb = float(st(b.to_numpy(float)[None, :])[0])
        out.append({"label": label, "strategy": va, "benchmark": vb, "difference": va - vb,
                    "ci_low": lo, "ci_high": hi, "fmt": fmt})
    return out


def _results(man: dict, tab: pd.DataFrame, s, n_boot: int, plots: list) -> str:
    key = man["key"]
    parts = [f"### {key}"]
    n = len(tab)
    q = s.quantiles
    rows = []
    for metric, label, fmt in (
        ("terminal_wealth", "terminal wealth (x start)", _num),
        ("max_drawdown_from_peak", "max drawdown from peak", _pct),
        ("excess_terminal_wealth", "excess terminal wealth vs benchmark", _num),
    ):
        sub = q[q["metric"] == metric]
        if sub.empty:
            continue
        rows.append([label, *[_with_ci(r.value, r.ci_low, r.ci_high, fmt)
                              for r in sub.itertuples()]])
    parts.append(
        f"Quantiles over {n} paths with {int(CI_LEVEL * 100)}% percentile-bootstrap intervals:\n\n"
        + _md_table(["metric", *[f"q{int(x * 100):02d}" for x in sorted(q["quantile"].unique())]],
                    rows)
    )

    bstats = benchmark_stats(man, tab, n_boot)
    if bstats is not None:
        brow = [[r["label"], r["fmt"](r["strategy"]), r["fmt"](r["benchmark"]),
                 _with_ci(r["difference"], r["ci_low"], r["ci_high"], r["fmt"])]
                for r in bstats]
        parts.append(
            f"Strategy against the benchmark ({man.get('benchmark') or 'none'}), same paths, "
            "same costs; the difference interval resamples paths jointly:\n\n"
            + _md_table(["statistic", "strategy", "benchmark", "difference [95% CI]"], brow)
        )
    else:
        parts.append("No benchmark was run for this scenario.")

    cv = s.cvar
    cv_rows = []
    for m, label, fmt in (("terminal_wealth", "terminal wealth (worst tail)", _num),
                          ("max_drawdown_from_peak", "max drawdown (deepest tail)", _pct)):
        c = cv[m]
        cv_rows.append([label, f"{c['alpha']:.0%}", _with_ci(c["value"], c["ci_low"],
                                                             c["ci_high"], fmt)])
    parts.append(
        "Tail means (CVaR):\n\n" + _md_table(["metric", "tail", "value [95% CI]"], cv_rows)
    )

    r, rec = s.ruin, s.recovery
    ruin_lvl = float(man.get("ruin_level", 0.5))
    parts.append(
        f"Ruin (equity at or below {ruin_lvl:.0%} of start at any bar): "
        f"{_pct(r['p'])} [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] "
        f"({r['n_ruined']} of {r['n']} paths, Wilson interval)."
    )
    ci_s, ci_m = rec["share_recovered_ci"], rec["median_time_ci"]
    parts.append(
        "Recovery of the worst drawdown from the all-time peak (Kaplan-Meier): share "
        f"recovered by the horizon {_pct(rec['share_recovered'])} "
        f"[{_pct(ci_s[0])}, {_pct(ci_s[1])}]; median time to recover "
        f"{_num(rec['median_time'], 0)} bars "
        f"[{_num(ci_m[0], 0)}, {_num(ci_m[1], 0)}] "
        f"(undefined when fewer than half recover); {rec['n_censored']} of {rec['n']} paths "
        f"censored at the {_num(rec['horizon_bars'], 0)}-bar horizon."
    )
    for title, name in plots:
        parts.append(f"![{title}]({name})")
    return "\n\n".join(parts)


def _variants(scen: list[tuple[dict, pd.DataFrame]], summaries: dict) -> str | None:
    live = [(m, t) for m, t in scen if not t.empty and m.get("axis_choices")]
    if len(live) < 2:
        return None
    stats = {}
    for m, t in live:
        stats[m["key"]] = {
            "tw": float(t["terminal_wealth"].median()),
            "dd": float(t["max_drawdown_from_peak"].median()),
            "ruin": float(summaries[m["key"]].ruin["p"]),
            "ex": float(t["excess_terminal_wealth"].median())
            if "excess_terminal_wealth" in t.columns else float("nan"),
        }
    rows = []
    for m, _ in live:
        s = stats[m["key"]]
        rows.append([_choices(m), _num(s["tw"]), _pct(s["dd"]), _pct(s["ruin"]),
                     _num(s["ex"])])
    parts = [
        "Each row is one scenario's own result. Differences between rows measure how much the "
        "answer depends on judgment calls (axis options), not on data. No figure is pooled or "
        "averaged across scenarios.",
        _md_table(["scenario", "median terminal wealth", "median max drawdown",
                   "P(ruin)", "median excess wealth"], rows),
    ]
    axes = list(dict.fromkeys(a for m, _ in live for a in m["axis_choices"]))
    arows = []
    for a in axes:
        opts = list(dict.fromkeys(m["axis_choices"].get(a) for m, _ in live))
        if len(opts) < 2:
            continue
        for o in opts:
            ks = [m["key"] for m, _ in live if m["axis_choices"].get(a) == o]
            tws = [stats[k]["tw"] for k in ks]
            dds = [stats[k]["dd"] for k in ks]
            arows.append([a, str(o), str(len(ks)),
                          f"{_num(min(tws))} to {_num(max(tws))}",
                          f"{_pct(min(dds))} to {_pct(max(dds))}"])
    if arows:
        parts.append(
            "Range of the scenario medians when one axis option is fixed (minimum to "
            "maximum over the other axes):\n\n"
            + _md_table(["axis", "option", "scenarios", "median terminal wealth",
                         "median max drawdown"], arows)
        )
    return "\n\n".join(parts)


SANITY_MIN_MOVE = 0.01  # moves smaller than 1% are not compared by sign


def path_sanity(man: dict, tab: pd.DataFrame) -> list[dict]:
    """Per asset: median crash-leg and horizon return across paths, the crash-leg return
    implied by the stress loadings on the anchored factors, and flags.

    ``expected`` is ``sum_f beta_stress[asset, f] * median_path(factor crash-leg return)``
    over the anchored factors. ``warning`` is set when the median crash-leg return has the
    opposite sign to ``expected`` (both at least 1%); ``imputed`` marks assets whose stress
    loadings were borrowed from neighbours.
    """
    L = man.get("loadings") or {}
    assets = man.get("assets") or L.get("assets") or []
    anchored = set((man.get("spec") or {}).get("anchors") or {})
    beta = None
    if L and L.get("regimes"):
        regime = L["regimes"][-1]
        beta = {
            a: dict(zip(L["factors"], L["betas"][regime][i], strict=True))
            for i, a in enumerate(L["assets"])
        }
    fmove = {}
    for f in anchored:
        col = f"{FACTOR_PREFIX}{f}"
        if col in tab.columns:
            fmove[f] = float(tab[col].median())
    imputed = set(L.get("imputed", []))
    imp = L.get("imputation") or {}
    rows = []
    for a in assets:
        c, h = f"{CRASH_PREFIX}{a}", f"{HORIZON_PREFIX}{a}"
        if c not in tab.columns:
            continue
        crash, hor = float(tab[c].median()), float(tab[h].median())
        expected = None
        if beta is not None and a in beta and fmove:
            expected = sum((beta[a].get(f) or 0.0) * m for f, m in fmove.items())
        warning = ""
        if (
            expected is not None and abs(expected) >= SANITY_MIN_MOVE
            and abs(crash) >= SANITY_MIN_MOVE and np.sign(crash) != np.sign(expected)
        ):
            warning = (
                f"WARNING: median crash-leg return {crash:+.1%} has the opposite sign of the "
                f"return implied by its stress loadings on the anchored factors "
                f"({expected:+.1%}); check the loadings, the neighbours and the shock direction"
            )
        rows.append({
            "asset": a, "crash_median": crash, "horizon_median": hor, "expected": expected,
            "warning": warning, "imputed": a in imputed,
            "amplification": (imp.get(a) or {}).get("amplification"),
        })
    return rows


def _sanity(scen: list[tuple[dict, pd.DataFrame]]) -> str:
    parts = [
        "Median across paths of each asset's cumulative return over the crash leg and over the "
        "whole horizon, next to the crash-leg return implied by its stress loadings on the "
        "anchored factors. A sign disagreement or an imputed asset deserves a look before the "
        "results are read."
    ]
    for man, tab in scen:
        if tab.empty:
            continue
        rows = path_sanity(man, tab)
        if not rows:
            parts.append(f"**{man['key']}**: per-asset path diagnostics are not available.")
            continue
        body = []
        for r in rows:
            note = []
            if r["imputed"]:
                amp = r["amplification"]
                note.append("IMPUTED (stress loadings borrowed from neighbours"
                            + (f", s={amp:.2f}" if amp is not None else "") + ")")
            if r["warning"]:
                note.append(r["warning"])
            body.append([r["asset"], _pct(r["crash_median"]), _pct(r["horizon_median"]),
                         "n/a" if r["expected"] is None else _pct(r["expected"]),
                         "; ".join(note) or "-"])
        parts.append(
            f"**{man['key']}**\n\n"
            + _md_table(["asset", "median crash-leg return", "median horizon return",
                         "implied by stress loadings", "flags"], body)
        )
    return "\n\n".join(parts)


# --- public ------------------------------------------------------------------------------------


def write_report(
    result_or_dir,
    out_dir,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    plots: bool = True,
) -> Path:
    """Write ``REPORT.md`` (and PNG plots when matplotlib is installed) into ``out_dir``.

    ``result_or_dir`` is the table returned by :func:`~crisispaths.runner.run_mc` /
    :func:`~crisispaths.runner.run_sweep`, or the runner's output directory.
    Returns the path of ``REPORT.md``.
    """
    src, scen = _load(result_or_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for man, tab in scen:
        if tab.empty:
            continue
        core = tab.drop(columns=[c for c in tab.columns if c.startswith(EXTRA_PREFIXES)])
        summaries[man["key"]] = summarize(
            core, alpha=alpha, n_boot=n_boot, seed=int(man["seed"]),
            analog_exact=bool(man.get("analog_exact")),
        )
    live = [(m, t) for m, t in scen if not t.empty]

    plt = _pyplot() if plots else None
    made: dict[str, list] = {}
    if plt is not None:
        for man, tab in live:
            made[man["key"]] = _plots(plt, tab, man, summaries[man["key"]], out)

    sections = [
        "# crisispaths scenario report",
        f"> **{BANNER}**",
        "## Caveats\n\n" + _caveats(scen, summaries),
        "## Scenario provenance\n\n" + _provenance(scen),
        "## Factor loadings\n\n" + _loadings(scen),
        "## Path sanity\n\n" + _sanity(scen),
    ]
    res = ["## Results"]
    if plots and plt is None:
        res.append(
            "Plots were skipped: matplotlib is not installed "
            "(`pip install crisispaths[report]`)."
        )
    for man, tab in live:
        res.append(_results(man, tab, summaries[man["key"]], n_boot, made.get(man["key"], [])))
    sections.append("\n\n".join(res))
    var = _variants(scen, summaries)
    if var:
        sections.append("## Variant comparison\n\n" + var)
    sections.append(
        f"---\nGenerated by crisispaths from `{src.name}`. Not financial advice; "
        "past crises do not repeat."
    )
    path = out / "REPORT.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


__all__ = ["BANNER", "intraday_warning", "path_sanity", "write_report"]

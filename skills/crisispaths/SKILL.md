---
name: crisispaths
description: Use when the user asks to stress-test, crisis-test or backtest a trading or portfolio strategy under crisis scenarios (AI bubble, US default, dollar debasement, market crash), asks "what happens to my strategy if ...", or wants a crisispaths REPORT.md, summary.json or PDF. Also use to write or verify commentary on a crisispaths report. Do not use for forecasting, return prediction or live trading.
---

# crisispaths skill

Run a strategy through simulated crisis scenarios and report the results as
conditional what-ifs. Full agent rules: `AGENTS.md` at the repository root.

## Workflow

1. **Inputs.** Need price levels (csv/parquet, date in the first column), the
   strategy as `strategy(prices) -> weights` in an importable module, and
   `--costs-bps`. Factor returns (`--factors`) make the run offline; otherwise
   the `[data]` extra and network are needed. If real data cannot be obtained,
   ask for CSV files or run `python examples/offline_demo.py 100 demo_out` and
   label every number synthetic.
2. **Adapt the strategy** (see AGENTS.md template): same-index weights, NaN = 0,
   no look-ahead, importable `module:fn`.
3. **Pick scenarios.** `crisispaths scenarios`, `crisispaths scenarios --family F`.
   Prefer a small sweep (`--family F --axis key=a,b`) when the user needs to see
   how much depends on a judgment call.
4. **Run.**
   `crisispaths run --prices P --factors F --strategy module:fn --scenario NAME --n-paths 500 --seed 0 --costs-bps C --out out`
   (`--costs-bps` mandatory; `--cost-multiplier` default 3; `--workers 1` in a sandbox).
5. **Read** `out/REPORT.md` and `out/summary.json`: banner, caveats, provenance
   with `calibrated: false` parameters, loadings (IMPUTED flags), path sanity
   (WARNINGs), results with intervals vs benchmark, variant comparison.
6. **Write commentary** (optional) from `summary.json` only, keeping the banner
   framing, then `crisispaths verify-commentary commentary.md --summary out/summary.json`
   and fix until it passes. Add `crisispaths report out --pdf --commentary commentary.md`
   for a PDF.

## Never

- Average or pool across scenarios, quote an expected return, call anything a
  forecast, or give a scenario a probability.
- Invent or round numbers beyond the displayed precision.
- Hide the caveats, the judgment parameters, an IMPUTED asset or a WARNING.
- Present results as paper or live trading performance.
- Run with zero costs without saying crisis costs are not zero.

## Check before answering

- Did every number come from `summary.json` / `REPORT.md`?
- Did the answer say results are conditional on the scenario and list the
  judgment parameters that drive them?
- Is the path count stated (200 minimum, 500+ for tails)?
- Did `verify-commentary` pass? (It checks traceability and framing, not truth.)

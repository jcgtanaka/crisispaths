# crisispaths

[![CI](https://github.com/jcgtanaka/crisispaths/actions/workflows/ci.yml/badge.svg)](https://github.com/jcgtanaka/crisispaths/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Backtest any strategy under simulated crisis scenarios. A scenario is a
declarative, composable spec; `crisispaths` turns it into thousands of
reproducible synthetic paths for your own assets, runs your strategy and a
benchmark on each, and writes a report that says how much of the answer is
judgment.

Status: alpha (0.1.0). The API may still change before 1.0; see
[docs/limitations.md](docs/limitations.md) for what the model does not do.

> Conditional on this scenario occurring. These are what-ifs, not forecasts.
> No probability is attached to any scenario.

## What it is

- A strategy-agnostic, asset-agnostic toolkit: your strategy is a function
  `prices -> weights`; your assets are whatever columns your price panel has.
- Scenario families with independent axes (dollar direction, aftermath, depth,
  crash speed, ...) so any pairing is expressible and every number that is a
  judgment call is flagged `calibrated: false` with a rationale.
- Frequency-agnostic through `bars_per_year` and `bars_per_day` (daily simulation,
  intraday disaggregation; see the frequency section below).
- Reproducible and resumable: path `i` of a seed regenerates alone, results
  are append-only files, parallel runs equal serial runs.
- Free data only (Yahoo Finance and FRED proxies, or your own factor panel);
  no pickles on disk.

## What it is not

- Not a forecasting tool. Scenarios are conditional what-ifs. The package
  never reports an expected return across scenarios.
- Not an execution or trading system, and not a market simulator at tick level.
- Not financial advice.

## Install

```bash
pip install -e .                 # core: numpy, pandas, scipy, pyyaml, platformdirs
pip install -e ".[report]"       # + matplotlib for the report plots
pip install -e ".[data]"         # + yfinance, pandas-datareader, pyarrow
pip install -e ".[pdf]"          # + reportlab, matplotlib for REPORT.pdf
pip install -e ".[dev]"          # tests, ruff, matplotlib, nbformat
```

Python 3.12 or newer.

**Experimental platforms.** Linux is the supported platform. Windows and macOS
are experimental: the author has not tested them, the parallel runner relies on
the `spawn` start method, and CI runs them as non-blocking jobs. Please report
issues.

## Quick start

```python
import crisispaths as cp
from crisispaths.runner import run_mc
from crisispaths.report import write_report

# prices: price levels (DatetimeIndex x assets); factors: factor returns (see below)
table = run_mc(
    prices, "my_strategies:sma_trend", "ai_bubble_worst",   # importable "module:fn"
    n_paths=200, seed=0, factors=factors, out_dir="out",
    costs_bps=5, cost_multiplier=3,                          # crisis costs are not zero
)
write_report(table, "out")        # out/REPORT.md (+ PNGs when matplotlib is installed)
```

Or from the shell (the same run, resumable if killed):

```bash
crisispaths scenarios                                    # aliases and families
crisispaths run --prices prices.csv --factors factors.csv --strategy my_strategies:sma_trend \
    --scenario ai_bubble_worst --n-paths 200 --costs-bps 5 --out out
```

`python examples/offline_demo.py` runs the whole workflow on a synthetic market
with no network.

## Scenario catalogue

Scenarios are built from a family: a base scenario plus independent axes, each
with named options. Named presets such as `ai_bubble_worst` or
`us_bankruptcy_haven_v_recovery_history_consistent` are aliases for one
`compose` call (`cp.list_scenarios()`).

**us_bankruptcy** (grid: 486 combinations)

| Axis | Options (default first) |
|---|---|
| complexity | `full` (EQ, USD, HAVEN, RATES, CREDIT, VOL), `standard` (EQ, USD, HAVEN), `minimal` (EQ) |
| usd | `neutral` (USD unanchored), `debasement` (USD -20..-10%, HAVEN +15..30%/yr after liquidation), `haven` (USD +5..+10%, HAVEN -5..0%/yr) |
| aftermath | `prolonged_bear` (-20%/yr), `v_recovery` (50-80% over 9M), `stagnation` |
| rates_credit | `shock` (RATES +100..+250bp, CREDIT +150..+350bp), `history_consistent` (unanchored); full complexity only |
| depth | `base` (median loss 40%), `mild` (25%), `severe` (55%) |
| crash_speed | `base` (3M), `fast` (1M), `slow` (6M) |

**ai_bubble** (grid: 144 combinations)

| Axis | Options (default first) |
|---|---|
| complexity | `standard` (EQ, TECH analogs), `minimal` (EQ), `full` (+ USD, RATES, VOL generated conditionally) |
| crash_source | `analog_exact`, `analog_bootstrap` (block bootstrap of 2000-2002), `parametric` (lognormal loss, IXIC -78% / SPX -49%) |
| aftermath | `analog`, `prolonged_bear`, `v_recovery` (labelled COUNTERFACTUAL), `stagnation` |
| tech_spread | `as_2000`, `amplified_1_5x` |
| depth | `as_2000`, `severe` (loss scaled by 1.25) |

`crisispaths scenarios --family F` prints the same table for any family
(including your own, loaded with `cp.load_family`).

### Judgment versus calibrated

Be clear about what each number is:

- **Calibrated** from your data and the history windows: factor loadings (per
  regime, with standard errors and shrinkage), residual volatility and
  correlation, the conditional-factor map, analog crash windows.
- **Judgment** (`calibrated: false`, always with a rationale): the size of a
  hypothetical shock (for example the 40% median equity loss in
  `us_bankruptcy`), the sign and size of USD, rates and credit moves that no
  historical window settles, the aftermath drift and recovery fractions, crash
  length, fat-tail degrees of freedom. These are assumptions, not estimates.

Assets with no data in the stress windows (for example BTC-USD in 2000-2002)
get imputed stress loadings: their calm loadings times ONE scalar `s`, fitted
by through-origin least squares over their nearest neighbours in calm-loading
space and clipped to `[0.5, 2.0]` (the clip bounds are a judgment). A calm sign
is never flipped, the imputed rows carry no standard errors, and the report
marks them `IMPUTED` with the `s` used.

**Calm drift.** After a V recovery and before the onset the generator
block-bootstraps calm-window returns. Raw calm returns carry each asset's
historical trend (a 2014-2025 bull run in BTC-USD reappears as a calm-period
gain in every path), which is not a property of the scenario. So
`calibrate(..., calm_drift="zero")` (the default) subtracts the mean of each
asset and factor column over its valid calm bars from the pool: volatility,
correlation and tails are kept, the trend is not. Use
`calm_drift="historical"` (CLI `--calm-drift historical`) to keep the raw
returns. Scenarios that never draw from the pool (an analog aftermath with
onset `0D`, `prolonged_bear`, `stagnation`) are bit-identical either way. The
choice is part of the calibration hash and the run pin, and the report
provenance prints it.

The report lists every judgment parameter of every scenario explicitly and,
for a sweep, shows how far the results move when a judgment axis changes. If
the spread between variants is large, the conclusion is a property of your
assumptions, not of the data.

## Composing scenarios

`compose` picks one option per axis (omitted axes use their default) and
returns a validated `ScenarioSpec` that records what was chosen.

Rules: axes apply in declaration order and a later axis wins on a conflicting
value; `rationale` strings accumulate and `calibrated` is `false` as soon as
any patch says so. An option whose `requires` is not met is skipped with a
`ScenarioWarning` (and listed in `spec.metadata["skipped"]`), never silently.
Every spec exposes `spec.key`, `spec.content_hash` and
`spec.metadata = {family, axis_choices, overrides, skipped}`.

```python
import crisispaths as cp

spec = cp.compose("us_bankruptcy")                          # defaults
spec = cp.compose("us_bankruptcy", usd="debasement", aftermath="v_recovery")
spec = cp.compose("us_bankruptcy", depth="severe",          # dotted-path overrides last
                  overrides={"anchors.EQ.params.median": 0.5, "crash_length": "6M"})

print(cp.grid_size("us_bankruptcy", usd=["debasement", "haven"], depth=["mild", "base"]))
for choices, spec in cp.enumerate_scenarios("us_bankruptcy", usd=["debasement", "haven"]):
    ...                                                      # lists are swept, scalars fixed

rep = cp.grid_report("us_bankruptcy", usd=["debasement", "haven", "neutral"],
                     complexity=["minimal", "standard", "full"])
print(rep.n_total, rep.n_unique)   # combinations that compose to identical content collapse

print(cp.describe_axes("ai_bubble")["axes"]["crash_source"]["options"])
mine = cp.load_family("my_family.yaml")                      # same validation as packaged ones
```

## Generating paths and backtesting

`calibrate` estimates everything once from your price panel and a factor
panel (or a `FactorProxySource`); `generate_path` draws one reproducible path
per call, with no network access.

```python
cal = cp.calibrate(prices, spec, clock=cp.Clock(252, "business"), factors=factors)
path = cp.generate_path(cal, spec, i=0, seed=42, clock=cp.Clock(252, "business"))
path.returns            # asset log returns, one row per synthetic bar
path.meta["anchors"]    # drawn cumulative targets per anchored factor
```

Model in one line: anchored factors follow their drawn targets, unanchored
factors follow `f_c = Gamma f_a + eta` (a stress-window regression, no
intercept, so they carry the historical crisis-conditional mean); assets load
on all factors plus t-copula residuals. Shock units: `pct` is a fraction
(`-0.20` is -20%), `percent` is points, `bp` is basis points, `log_change` is
a log change. Path `i` of a seed depends on nothing else, and the first `h`
bars of a horizon-`H` path equal the horizon-`h` path.

`cp.backtest(strategy, prices, scenario, ...)` is the single-process reference:
it composes, calibrates, generates `n_paths`, runs your strategy and an
equal-weight benchmark through the same harness, and returns `res.table`
(one row per path), `res.summary`, `res.spec` and `res.meta`.

**Strategy contract.** `strategy(prices) -> weights`. `prices` is your history
followed by the simulated path (compounded from your last real close, so
indicators are warm from the first simulated bar); `weights` has the same index
and columns that are a subset of the assets. NaN means zero and the remainder
is cash. Shorts are allowed; `leverage_cap` limits gross exposure. A weight set
from the close of bar `t` earns the return of bar `t + 1`. Weights are
rebalanced to target every bar; costs are
`costs_bps / 1e4 * cost_multiplier * sum(|change in weight|)`. A bar at or below
-100% is clipped, flagged as ruin and equity stays at zero. `mode="returns"`
lets the callable return strategy returns instead; there you own the timing and
the look-ahead check cannot audit it.

**Look-ahead check.** `check_lookahead` reruns the strategy on prefixes of the
history and requires the weights on the shared prefix to match the full run.
`backtest` and the runner run it once and raise `LookaheadError`.

**Benchmark.** The default benchmark is equal weight, rebalanced to 1/N every
bar, through the same harness, path and costs. It is a rebalanced benchmark,
not buy-and-hold: it harvests rebalancing gains and pays rebalancing costs,
which matters when the assets diverge strongly in a scenario. Pass
`benchmark=` to use your own, or `None` to skip it.

**Metrics** (times in bars, drawdowns positive fractions):

| Column | Meaning |
|---|---|
| `terminal_wealth` | final equity divided by start equity |
| `terminal_log_return` | its log (floored at 1e-12 for a wiped-out path) |
| `max_drawdown_from_start` | worst fall from the running peak inside the simulated segment |
| `max_drawdown_from_peak` | worst fall from the all-time peak, including the history peak of the strategy's own equity |
| `loss_from_start` | `1 - min(equity) / start`, floored at 0 |
| `time_to_trough_bars` | bars from segment start to the trough of the worst drawdown from peak |
| `time_to_recover_bars`, `recovered`, `recovery_time_bars` | bars from that trough until equity regains the peak it fell from |
| `ruined`, `time_to_ruin_bars` | equity ever at or below `ruin_level` (default 50%) of start |
| `excess_terminal_wealth`, `excess_max_drawdown_from_peak` | strategy minus the benchmark; a positive drawdown difference means the strategy fell further |

**Censoring.** A path that never regains its peak inside the horizon is
censored: `recovered=False`, `time_to_recover_bars` is NaN and
`recovery_time_bars` is the time observed after the trough. Summaries never
average these times; they report the Kaplan-Meier share recovered by the
horizon and the Kaplan-Meier median (NaN when fewer than half recover).

## Running at scale: runner, sweeps, CLI

`run_mc(prices, strategy, specs, *, n_paths, seed, out_dir, workers, costs_bps,
cost_multiplier, resume, ...)` is the resumable parallel runner.

- `strategy` is an importable `"module:fn"` string. It is required when
  `workers > 1` because workers use the `spawn` start method on every OS (a
  top-level function of an importable module is converted automatically;
  lambdas and functions defined in `__main__` are rejected with a clear error).
- `workers=None` uses `min(cpu_count - 1, n_paths)`; `workers=1` runs in
  process. Rows are written in path order, so the output does not depend on
  `workers`.
- Per scenario `<key>` the output directory holds `results_<key>.csv` (one row
  per path, appended with `O_APPEND` and `fsync`; a partial last line from a
  kill is discarded), `pin_<key>.json` (spec content hash, calibration hash,
  seed, package version, costs and other result-changing parameters) and
  `manifest_<key>.json` (descriptive data for the report).
- Resume: finished path indices are read from the CSV and skipped. A pin that
  does not match raises `PinMismatchError`: runs are never silently mixed. Use
  `resume=False` (`--no-resume`) to start over. `KeyboardInterrupt` keeps the
  partial results; the same call resumes.
- `run_sweep(prices, strategy, family, axes, out_dir=..., max_scenarios=64)`
  enumerates scenarios with `dedupe=True`, runs each and returns one long table
  with the axis choices as columns. It refuses sweeps with more distinct
  scenarios than `max_scenarios` and prints the grid numbers.

```bash
crisispaths scenarios                        # aliases and families
crisispaths scenarios --family us_bankruptcy # axes, options, defaults
crisispaths scenarios --grid us_bankruptcy --axis usd=debasement,haven \
    --axis aftermath=prolonged_bear,v_recovery

crisispaths run --prices prices.parquet --factors factors.parquet \
    --strategy my_strategies:sma_trend \
    --family us_bankruptcy --axis complexity=full --axis usd=debasement,haven \
    --axis aftermath=prolonged_bear,v_recovery \
    --override anchors.EQ.params.median=0.5 \
    --n-paths 500 --seed 0 --costs-bps 5 --cost-multiplier 3 --workers 4 --out out

crisispaths report out                       # rebuild out/REPORT.md
```

`--costs-bps` is required (crisis costs are not zero; pass `0` on purpose to
run without). `--cost-multiplier` defaults to 3. `--prices` and `--factors` are
csv or parquet with the date as the first column; without `--factors` the
proxies are downloaded (needs the `[data]` extra and network). Exit codes: 0
success, 1 runtime error (bad file, unknown scenario, pin mismatch, sweep
guard), 2 usage error, 130 interrupted (resumable).

## Summary JSON, PDF and commentary check

- `summary.json` (`schema_version` 1, schema in
  `src/crisispaths/schema/summary.schema.json`) is written by `run` and by
  `crisispaths report DIR --json`: provenance, judgment parameters, loadings,
  path sanity, metrics with intervals, the benchmark comparison and, for sweeps,
  per-scenario variant rows and per-axis-option ranges. It never contains a
  figure aggregated across scenarios.
- `crisispaths report DIR --pdf` renders REPORT.md (same section order, charts
  embedded) to `REPORT.pdf`; needs the `[pdf]` extra (exit code 2 with a clear
  message otherwise). `--commentary FILE.md` appends commentary last under
  "AI-generated commentary (not part of the deterministic report)" after
  `verify-commentary` passes; `--no-verify` stamps it UNVERIFIED.
- `crisispaths verify-commentary FILE.md --summary summary.json` is a
  heuristic check of traceability (every decimal, percentage or ratio must match
  a value in summary.json at the displayed precision), framing (no expected
  return, averages across scenarios, forecasts, scenario probabilities, "safe",
  "guaranteed", paper or live performance claims) and conditionality. It checks
  traceability and framing, NOT semantic truth: a number can coincide with an
  unrelated value, and a fluent false sentence passes. Exit 0 pass, 1 fail,
  2 usage.

## Use with an LLM agent (Claude Code, Codex)

The repository carries `AGENTS.md` (rules and command sequence) and
`skills/crisispaths/SKILL.md` (the same workflow as an agent skill).
A worked transcript is in `examples/agent_walkthrough.md`. Copy-paste prompt:

```text
Use the crisispaths skill (skills/crisispaths/SKILL.md, rules in AGENTS.md).
Stress-test the strategy in my_strategies.py against ai_bubble_worst and the
us_bankruptcy debasement/haven x prolonged_bear/v_recovery grid using
prices.csv and factors.csv, 500 paths, costs 5 bps with the default crisis
multiplier. Adapt the strategy to strategy(prices) -> weights if needed, run
crisispaths, then write a short commentary from summary.json only. Keep the
banner and caveats, list the calibrated:false parameters, mention every IMPUTED
asset and path sanity WARNING, never average across scenarios or call anything
a forecast, and run verify-commentary until it passes. If the data download is
impossible, tell me and use the offline demo instead.
```

## Report anatomy

`REPORT.md` has a fixed order:

1. **Banner**: "Conditional on this scenario occurring. These are what-ifs, not
   forecasts. No probability is attached to any scenario."
2. **Caveats**: analog-exact conditionality of the intervals, number of paths,
   censoring count, intraday optimism warning (more than 252 bars per year with
   high turnover), the cost multiplier used.
3. **Scenario provenance**: family, axis choices, overrides, content hash,
   seed, version, and every `calibrated: false` parameter listed as JUDGMENT.
4. **Factor loadings**: observations, standard errors, shrinkage share,
   unaffected and imputed flags (with the amplification `s`) and warnings.
5. **Path sanity**: per asset, the median crash-leg and horizon return across
   paths against the crash-leg return implied by its stress loadings on the
   anchored factors; a sign disagreement is a WARNING, imputed assets are
   flagged.
6. **Results**: quantiles, strategy against benchmark with bootstrap
   intervals, tail means, ruin (Wilson), Kaplan-Meier recovery; plots
   (equity percentile fan, drawdown histogram, ten worst paths, recovery curve)
   when matplotlib is installed.
7. **Variant comparison** (sweeps only): each scenario's own numbers and the
   range per axis option, so you see how much is judgment. Nothing is pooled
   or averaged across scenarios.

## Frequency and intraday bars

Paths are ALWAYS simulated at the calibration frequency (daily). Your bar
clock is reached afterwards:

- **Intraday bars.** Pass intraday prices to `calibrate()` (or `run_mc` /
  `backtest`): the per-slot variance profile of the session is estimated
  (U-shape, overnight-gap share, Student-t degrees of freedom), prices are
  aggregated to daily for calibration, and each simulated daily return is
  split into the session's bars by a Brownian bridge. The bars of a session
  sum exactly to the daily return, so re-aggregating an intraday path gives the
  daily path. `Clock(calendar=prices.index)` infers `bars_per_day`,
  `is_intraday` and the overnight gap; `time_to_recover_days` is reported next
  to the bar counts.
- **No intraday history.** With daily prices and an intraday `Clock` a flat
  profile is used and a `UserWarning` says so.
- **Coarser bars.** A weekly or monthly clock on a daily calibration sums
  daily log returns in blocks (the clock must be an integer fraction of the
  calibration frequency).
- Memory: one path is built in chunks of 64 sessions; one 5-minute path of 10
  assets over 504 sessions (39,312 bars) peaks near 11 MiB.

Honest limits of intraday results (also printed in every intraday report):
inside a session the path is a bridge between simulated daily closes, so
there are no jumps inside the session, no trading halts or liquidity gaps, the
cost model is flat per unit of turnover, the Epps effect is not modelled (the
daily residual correlation is applied to every bar), and strategies that react
to intraday moves (stop-losses, intraday breakouts) look better than they
would in reality. Minute bars are never simulated directly: it would wash out
the fat daily crash tails (central limit theorem) and the memory would explode.

## Limitations and prior art

Limitations:

- Scenarios are a small set of hand-written families; the shock sizes marked
  JUDGMENT are assumptions.
- `analog_exact` replays one historical window in every path, so its intervals
  are conditional on that single history and understate uncertainty.
- Loadings are estimated on few stress bars and shrunk; assets flagged
  unaffected are a model claim, not a fact.
- No market impact, liquidity or financing model beyond proportional costs and
  a cash rate; no intraday microstructure (intraday bars are a bridge between
  simulated daily closes).
- Equal-weight, rebalanced benchmark by default (see above).

Related projects worth knowing (this package is not a replacement for any of
them):

- [skfolio](https://github.com/skfolio/skfolio): portfolio optimisation and
  risk management on scikit-learn, including scenario-style stress tests.
- [qfwp](https://github.com/MarcosCarreira/qfwp): companion package with real
  historical episodes and a synthetic market.
- [fin-stresstest](https://github.com/LuisHCalderon/StressTest): financial
  stress scenario analysis.
- [fsynth](https://github.com/welcra/fsynth): synthetic financial data with
  Heston stochastic volatility and jump diffusion.
- [py-vallocation](https://github.com/enexqnt/py-vallocation): portfolio
  allocation with views and robust optimisation.
- [rustybt](https://github.com/jerryinyang/rustybt): backtesting engine built
  on Zipline-Reloaded.

## Disclaimer

crisispaths is research software. It is not financial advice. Scenarios are
what-ifs, not forecasts, and no probability is attached to any of them. Past
crises do not repeat; a strategy that survives these scenarios may fail in the
next real one.

## Development

```bash
ruff check .
pytest            # offline; `pytest -m network` runs the real-data example smoke test
```

See [docs/methodology.md](docs/methodology.md), [docs/limitations.md](docs/limitations.md),
[CONTRIBUTING.md](CONTRIBUTING.md) to add a family or an axis and
[CHANGELOG.md](CHANGELOG.md) for changes.

## License

MIT

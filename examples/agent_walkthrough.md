# Agent walkthrough (offline demo)

A worked command sequence for an LLM agent, run on the offline demo (synthetic
data, no network). The output below was captured from a real run with 50 paths;
long scenario names are shortened where marked. The numbers describe a made-up
market and mean nothing about real assets. Rules for agents: `AGENTS.md`.

## 1. Discover scenarios

```console
$ python -m crisispaths scenarios --family ai_bubble
family: ai_bubble
An AI-led equity bubble deflates, built from independent axes for model complexity, crash source, aftermath, technology spread and depth.

axis complexity (default: standard)
  Which factors are anchored; the others are generated conditionally.
  - minimal: Anchors EQ (S&P 500 analog) only.
  - standard [default]: Anchors EQ and TECH (Nasdaq residual analog).
  - full: EQ and TECH anchored; USD, RATES and VOL generated conditionally.

axis crash_source (default: analog_exact)
  Where the crash leg of the anchored factors comes from.
  - analog_exact [default]: Replay the realised 2000-2002 path.
  ...
```

## 2. Run the offline demo (a 2x2 sweep, 50 paths, costs 5 bps x 3)

```console
$ python examples/offline_demo.py 50 demo_out
                          median_terminal_wealth  p_ruin  median_max_drawdown
scenario
ai_bubble_...aftermath.analog-...-depth.as_2000                0.946     0.0                0.530
ai_bubble_...aftermath.analog-...-depth.severe                 0.936     0.0                0.540
ai_bubble_...aftermath.prolonged_bear-...-depth.as_2000        0.924     0.0                0.538
ai_bubble_...aftermath.prolonged_bear-...-depth.severe         0.916     0.0                0.546
report: demo_out/REPORT.md
```

Wall time on one core: about 7 s. 50 paths is below the 200-path minimum for tail
statistics; the report says so in its caveats.

## 3. Regenerate the machine-readable summary

```console
$ python -m crisispaths report demo_out --json
report: demo_out/REPORT.md
summary: demo_out/summary.json
```

`summary.json` (schema_version 1) holds, per scenario, the provenance, the
`calibrated: false` parameters, loadings, path sanity, metrics with intervals and
the benchmark comparison. It never holds a figure aggregated across scenarios.

## 4. Read REPORT.md in order

The first lines are the banner and the caveats; keep both in anything you quote:

```text
> **Conditional on this scenario occurring. These are what-ifs, not forecasts. No probability is attached to any scenario.**

## Caveats

- Scenarios are conditional what-ifs. Nothing here says a scenario will happen or how likely it is.
- Costs: costs_bps=5.0, cost_multiplier=3.0; effective 15 bps per unit of turnover.
...
  - only 50 paths (< 200): quantile, CVaR and recovery intervals are unstable; use at least 200 paths for tail statistics.
  - analog exact mode: the crash replays one historical window in every path, so the intervals are conditional on that single history and understate scenario uncertainty.
  - censoring: 50 of 50 paths did not recover within the horizon (censored); recovery is estimated with Kaplan-Meier, never a mean of observed times.
```

## 5. Write commentary from summary.json only, then verify it

`good.md`:

```text
Conditional on each scenario occurring, and on the scenario assumptions flagged JUDGMENT in
the provenance section, the SMA filter ended with a median terminal wealth of 0.946 of the
starting equity in the analog-aftermath variant and 0.924 in the prolonged-bear variant.
No path was ruined in either (0.0%). In 50 of 50 paths the worst drawdown had not recovered by
the horizon, so recovery is censored. This is a what-if exercise on synthetic data with only
50 paths per scenario; the intervals are unstable and nothing here is a forecast.
```

```console
$ python -m crisispaths verify-commentary good.md --summary demo_out/summary.json
verify-commentary: PASS
Heuristic check of traceability and framing only; it does not establish that the text is true.
$ echo $?
0
```

`bad.md` (an invented probability, an expected return, an untraceable number):

```text
The expected return of the filter is 4.7% across the grid, so it is safe. The probability of
an AI bubble is 12%, and the median terminal wealth is 1.31 in the base case.
```

```console
$ python -m crisispaths verify-commentary bad.md --summary demo_out/summary.json
verify-commentary: FAIL
  untraceable number '12%' in: The probability of an AI bubble is 12%, and the median terminal wealth is 1.31 in the base case.
  untraceable number '1.31' in: The probability of an AI bubble is 12%, and the median terminal wealth is 1.31 in the base case.
  forbidden framing [expected return] in: The expected return of the filter is 4.7% across the grid, so it is safe.
  forbidden framing [safe] in: The expected return of the filter is 4.7% across the grid, so it is safe.
  forbidden framing [scenario probability] in: The probability of an AI bubble is 12%, and the median terminal wealth is 1.31 in the base case.
  error: 2 number(s) cannot be traced to summary.json
  error: 3 forbidden framing(s)
  error: the commentary never acknowledges that results are conditional on the scenario (expected a phrase such as 'conditional on' or 'what-if')
Heuristic check of traceability and framing only; it does not establish that the text is true.
$ echo $?
1
```

Note what the verifier did NOT catch: `4.7%` happened to coincide with some value in
`summary.json`, so it was not flagged as untraceable. The check is heuristic (a
number only has to match some value at the displayed precision); the framing rules
caught the sentence anyway. A human still has to read the commentary.

## 6. Build the PDF

```console
$ python -m crisispaths report demo_out --pdf --commentary good.md
report: demo_out/REPORT.md
pdf: demo_out/REPORT.pdf
$ python -m crisispaths report demo_out --pdf --commentary bad.md
report: demo_out/REPORT.md
verify-commentary: FAIL
  ...
crisispaths: error: commentary failed verification; no PDF written (fix it, or pass --no-verify to stamp it UNVERIFIED)
```

The PDF has the same section order as REPORT.md, embeds the charts and appends
the verified commentary last under "AI-generated commentary (not part of the
deterministic report)".

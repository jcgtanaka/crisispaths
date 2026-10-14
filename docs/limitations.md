# Limitations

Read this before relying on any number from crisispaths.

## What the output is

Every result is conditional on a scenario occurring. Scenarios are what-ifs
written by people; no probability is attached to any of them and the package
never reports an expected return across scenarios. Past crises do not repeat.
This is research software, not financial advice.

## Model limits

- **Few, hand-written scenarios.** The packaged families (`us_bankruptcy`,
  `ai_bubble`) encode specific stories. Numbers marked JUDGMENT are assumptions
  with a stated rationale, not estimates.
- **Few stress bars.** Stress loadings come from a handful of historical
  windows. They are shrunk toward calm loadings and can be unreliable; the
  report prints observations, standard errors and shrinkage.
- **Imputation.** Assets with no stress data get calm loadings times one scalar
  `s` in `[0.5, 2.0]` borrowed from neighbours. This is a modelling claim; such
  assets are flagged IMPUTED.
- **Unaffected assets** are a model claim from an F-test, not a fact.
- **analog_exact** replays one historical window in every path, so its
  intervals are conditional on that single history and understate uncertainty.
- **Calm drift.** By default calm bootstrap rows are de-trended; the old
  behaviour is `calm_drift="historical"`.
- **Costs and liquidity.** Proportional costs and a cash rate only. No market
  impact, borrow constraints, margin calls, financing spreads or halts.
- **Benchmark.** The default benchmark is equal weight rebalanced every bar. It
  harvests rebalancing gains under high volatility and can exceed 1 in a crash
  scenario for that reason; pass your own benchmark when that matters.
- **Residual dynamics.** There is no GARCH-type residual filter in 0.1
  (planned); residual volatility follows the calibrated level and the scenario
  envelope.

## Frequency limits

- Simulation is daily. Intraday bars are a Brownian bridge between simulated
  daily closes: no jumps inside a session, no halts or liquidity gaps, flat cost
  model, no Epps effect (the daily residual correlation is applied to every
  bar). Stop-loss and intraday-breakout strategies look better than reality.
- Without intraday history the intraday profile is flat (a warning is issued).
- Sessions that cross midnight (some futures) are assigned by calendar date.
- Coarser clocks must be an integer fraction of the calibration frequency (within
  10%). Blocks have `k = round(252 / bars_per_year)` days, so a weekly clock sums
  5-day blocks (an effective 50.4 bars per year) and a trailing partial block is
  dropped.

## Platform limits

- Linux is supported. Windows and macOS are experimental: untested by the author
  and dependent on the `spawn` start method; please report issues.
- Continuous integration runs the full suite on Linux (Python 3.12 and 3.13) and on
  Windows, where it passes. On macOS the suite passes except for tests that compare
  exact random realisations (skipped there).
- Reproducibility: the same seed gives the same paths on one platform and library
  stack. The calibration is cross-platform stable, but the random draws are not
  guaranteed to be identical across operating systems, BLAS builds or numpy
  versions (matrix square roots depend on LAPACK). Realisations then differ while
  the distribution is the same, so summary statistics agree within sampling
  error. Pin your environment when you need exact reruns.
- Python 3.12 or newer only.

## Commentary verification limits

`verify-commentary` checks that numbers trace to some value in summary.json and
that no forbidden framing is used. It does not check that the text is true or
that a number is attached to the right quantity, and a number can coincide with
an unrelated value.

## Engineering limits

- Results files are append-only CSV; changing the package version, spec,
  calibration or run parameters requires `--no-resume` or a new output
  directory (the pin refuses to mix runs).
- Workers use the `spawn` start method; strategies must be importable
  `module:fn` strings for `workers > 1`.
- Free data proxies (Yahoo Finance, FRED) can change between downloads; freeze
  your inputs when you need bit-identical reruns.

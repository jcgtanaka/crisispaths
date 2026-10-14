# Changelog

## 0.1.0

First alpha release.

### Platform and tooling

- Python 3.12 and 3.13 only (`requires-python >= 3.12`, ruff target py312).
- Linux is supported; Windows and macOS are EXPERIMENTAL (untested by the
  author, spawn-based runner). CI runs them as non-blocking jobs.
- `summary.json` (`schema_version` 1, packaged JSON Schema), `report --pdf`
  (`[pdf]` extra), `verify-commentary`, `AGENTS.md`, `skills/crisispaths`,
  `examples/agent_walkthrough.md`.

### Added

- Scenario specs (YAML), clock, free-data factor proxies, factor loadings with
  shrinkage, path generation (anchored factors, conditional factors, t-copula
  residuals, analog replay and block bootstrap).
- Composable scenario families (`us_bankruptcy`, `ai_bubble`) with independent
  axes, dotted-path overrides, grid reports and de-duplicated sweeps.
- Strategy harness, look-ahead audit, path metrics with censored recovery
  (Kaplan-Meier), bootstrap and Wilson intervals, `backtest()`.
- Resumable parallel runner `run_mc` and sweep runner `run_sweep` (spawn
  workers, append-only results, pin files, partial-line recovery).
- `write_report`: REPORT.md with banner, caveats, provenance, loadings, path
  sanity, results and variant comparison; PNG plots with the `[report]` extra.
- `crisispaths` command line (`scenarios`, `run`, `report`).
- Intraday disaggregation: `Clock.bars_per_day`, `is_intraday`, `sessions`,
  `daily()`; `estimate_intraday_profile`, `disaggregate` (Brownian bridge),
  `aggregate_log_returns`; `time_to_*_days` metrics; weekly and monthly clocks
  by summing daily log returns.
- Examples (offline demo, real-data SMA example, notebook), docs
  (`docs/methodology.md`, `docs/limitations.md`), `RELEASING.md`, `py.typed`.

### Changed

- The calm bootstrap pool is de-trended by default (`calm_drift="zero"`); pass
  `calm_drift="historical"` (CLI `--calm-drift historical`) for the old
  behaviour. Results of scenarios that use the pool change; others do not.
- `backtest()` shares `run_one_path` with the runner.
- Reports state the frequency, bars per year, the intraday profile source and
  the intraday limits.

### Fixed

- Stress loadings of assets with no data in the stress windows are the calm
  loadings times one pooled, clipped scalar (was a per-factor ratio of
  neighbours that could flip a sign).
- Calibration with NaN-padded assets no longer emits numpy RuntimeWarnings.

### Known limitations

- Daily simulation only; intraday bars are a bridge between daily closes (no
  intra-session jumps, halts, liquidity gaps or Epps effect).
- Scenario shocks marked JUDGMENT are assumptions; stress loadings rest on few
  bars and are shrunk.
- analog_exact intervals are conditional on one replayed history.
- No GARCH residual filter, market impact or financing model.
- Free-data proxies can change between downloads.

### Planned

- GARCH-type residual volatility filter.
- Additional scenario families.

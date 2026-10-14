# AGENTS.md

Instructions for LLM coding agents (Claude Code, Codex, ...) that use or modify
crisispaths. The same workflow in skill form: `skills/crisispaths/SKILL.md`.

## What this tool is

crisispaths backtests a strategy under simulated crisis scenarios. Every result
is a conditional what-if. It is not a forecast and no probability is attached to
any scenario.

## Install and verify

```bash
python -m pip install -e ".[dev]"      # Python >= 3.12; add ,pdf for PDF output
ruff check .
pytest                                  # offline; ~2-3 minutes
```

## The command sequence for a user request

```bash
# 1. discover scenarios
crisispaths scenarios
crisispaths scenarios --family us_bankruptcy

# 2. run (prices.csv / factors.csv: date in the first column)
crisispaths run --prices prices.csv --factors factors.csv \
    --strategy my_strategies:my_strategy \
    --scenario ai_bubble_worst --n-paths 500 --seed 0 \
    --costs-bps 5 --cost-multiplier 3 --workers 1 --out out

# 3. rebuild artefacts if needed
crisispaths report out --json --pdf

# 4. commentary (if you write one): verify and fix until it passes
crisispaths verify-commentary commentary.md --summary out/summary.json
crisispaths report out --pdf --commentary commentary.md
```

Outputs in `out/`: `REPORT.md`, `summary.json` (schema_version 1), PNG charts,
`results_*.csv`, `pin_*.json`, `manifest_*.json`. `REPORT.pdf` with `--pdf`.

## Rules (not optional)

- `--costs-bps` is mandatory. Crisis costs are not zero. Keep the default
  `--cost-multiplier 3` unless the user says otherwise.
- Never average across scenarios, never report an expected return, never
  describe a result as a forecast or give a scenario a probability.
- Always keep the banner and the caveats block in anything you show the user.
- Always list the `calibrated: false` (JUDGMENT) parameters of the scenarios you
  report, and say which results depend on them.
- Mention every `IMPUTED` asset and every path sanity WARNING.
- Never invent numbers: quote only values in `summary.json` / `REPORT.md`, at the
  displayed precision. Use at least 200 paths (500+ for tails) and say how many
  were used.
- Do not describe results as paper or live trading performance.
- Run `verify-commentary` on any prose you write and fix it until it passes. It
  checks number traceability and framing only, not whether the text is true;
  you still have to read it against the report.
- A killed run resumes with the same command. A pin mismatch means a different
  run: use a new `--out` or `--no-resume`; never edit the pin files.

## Sandbox and no-network behaviour

- The tool itself needs no network when you pass `--factors` (a factor-return
  CSV or parquet). Without `--factors` it downloads proxies from Yahoo Finance
  and FRED (needs `pip install -e ".[data]"` and network).
- If the real-data download is impossible: ask the user for CSV files of prices
  (and factors), or run the offline demo to demonstrate the workflow on synthetic
  data: `python examples/offline_demo.py 100 demo_out`. Say clearly that the
  numbers are synthetic and mean nothing about real assets.
- Use `--workers 1` in constrained sandboxes. For `--workers > 1` the strategy
  must be an importable `module:fn` string (spawn start method; Windows and
  macOS are experimental).

## Adapting a user's strategy to `strategy(prices) -> weights`

`prices` is the history followed by the simulated path (DatetimeIndex, one
column per asset). Return target weights with the same index and columns that
are a subset of the assets. A weight set at the close of bar `t` earns bar
`t + 1`. NaN means 0 and the remainder is cash. Use no future data (the
look-ahead audit raises `LookaheadError` otherwise).

```python
# my_strategies.py  (must be importable from the working directory)
import pandas as pd


def my_strategy(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-capital trend filter: hold an asset while above its 150-bar average."""
    ma = prices.rolling(150).mean()
    held = (prices > ma).astype(float).where(ma.notna())   # NaN until warm
    return held / prices.shape[1]                          # 1/N per asset, rest cash
```

Put signal logic in this function only; keep sizing in weights, not in
position counts. Costs and leverage limits (`leverage_cap`) are applied by the
harness.

## Working on the code

TDD: failing test first. Conventional commits; no AI attribution lines. Keep
English only, no em dashes, no absolute paths or private references in files.
See `CONTRIBUTING.md`, `docs/methodology.md` and `docs/limitations.md`.

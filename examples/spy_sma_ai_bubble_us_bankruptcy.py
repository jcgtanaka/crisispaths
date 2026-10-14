"""SPY/TLT/GLD/BTC-USD with an SMA150 filter, stressed by two crisis families.

Needs the ``[data]`` extra and network access (Yahoo Finance and FRED):

    pip install -e ".[data,report]"
    python examples/spy_sma_ai_bubble_us_bankruptcy.py 200 out_spy

Runs ``ai_bubble`` at ``standard`` complexity and the ``us_bankruptcy`` 2x2
(``usd`` x ``aftermath``) at ``complexity=full``, then writes one REPORT.md per
family into ``out_dir/<family>``. A scenario is a conditional what-if; none of
the numbers below is a forecast and no probability is attached to any of them.

Assets that start later than SPY (GLD 2004, BTC-USD 2014) are NaN before
their first price; the strategy treats a missing price as "not held".
"""

import sys
from pathlib import Path

import pandas as pd

TICKERS = ["SPY", "TLT", "GLD", "BTC-USD"]
START = "1999-01-01"
SMA_WINDOW = 150


def sma150(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-capital share in each asset trading above its 150-bar average, else cash."""
    ma = prices.rolling(SMA_WINDOW).mean()
    return (prices > ma).astype(float).where(ma.notna()) / prices.shape[1]


def download_prices(start: str = START) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, progress=False, auto_adjust=True)["Close"]
    px = raw[TICKERS].astype(float)
    # SPY sets the (business-day) calendar; BTC-USD trades on weekends and is sampled on it
    return px.loc[px["SPY"].dropna().index]


def main(n_paths: int = 200, out_dir: str | Path = "out_spy", workers: int | None = None):
    import crisispaths as cp
    from crisispaths.data import YFinanceFredSource
    from crisispaths.report import write_report
    from crisispaths.runner import run_sweep

    prices = download_prices()
    source = YFinanceFredSource()
    clock = cp.Clock(calendar=prices.index)
    out = Path(out_dir)
    strategy = f"{Path(__file__).stem}:sma150"  # importable by spawned workers
    runs = {
        "ai_bubble": {"complexity": "standard"},
        "us_bankruptcy": {
            "complexity": "full",
            "usd": ["debasement", "haven"],
            "aftermath": ["prolonged_bear", "v_recovery"],
        },
    }
    heads = []
    for family, axes in runs.items():
        table = run_sweep(
            prices, strategy, family, axes, n_paths=n_paths, seed=0, clock=clock,
            source=source, out_dir=out / family, workers=workers, costs_bps=5.0,
            cost_multiplier=3.0,
        )
        write_report(table, out / family)
        heads.append(table.groupby("scenario", sort=False).agg(
            median_terminal_wealth=("terminal_wealth", "median"),
            median_benchmark_wealth=("bench_terminal_wealth", "median"),
            p_ruin=("ruined", "mean"),
            median_max_drawdown=("max_drawdown_from_peak", "median"),
            share_recovered=("recovered", "mean"),
        ))
    head = pd.concat(heads)
    print(head.round(3).to_string())
    print(f"reports under {out.resolve()}")
    return head


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200,
         sys.argv[2] if len(sys.argv) > 2 else "out_spy")

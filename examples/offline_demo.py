"""Offline demo: a trend filter under two variants of an AI-bubble crash.

Everything is synthetic (see ``synthetic_market.py``), so it runs anywhere and
is executed by the test suite with a small ``n_paths``.

    python examples/offline_demo.py            # 100 paths, writes ./demo_out
    python examples/offline_demo.py 300 out    # 300 paths into ./out

The numbers describe a made-up market and mean nothing about real assets; the
point is the workflow: sweep, resume-safe run, REPORT.md.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_market import make_market  # noqa: E402

import crisispaths as cp  # noqa: E402
from crisispaths.report import write_report  # noqa: E402
from crisispaths.runner import run_sweep  # noqa: E402
from crisispaths.summary import write_summary  # noqa: E402


def sma_filter(prices: pd.DataFrame, window: int = 150) -> pd.DataFrame:
    """Hold an asset (equal share of capital) while it trades above its moving average."""
    ma = prices.rolling(window).mean()
    return (prices > ma).astype(float).where(ma.notna()) / prices.shape[1]


def main(n_paths: int = 100, out_dir: str | Path = "demo_out", workers: int | None = 1) -> Path:
    prices, factors = make_market()
    table = run_sweep(
        prices, sma_filter, "ai_bubble",
        {"aftermath": ["analog", "prolonged_bear"], "depth": ["as_2000", "severe"]},
        n_paths=n_paths, seed=0, clock=cp.Clock(252, "business"), factors=factors,
        out_dir=out_dir, workers=workers, costs_bps=5.0, cost_multiplier=3.0,
    )
    report = write_report(table, out_dir)
    write_summary(table, out_dir)
    head = table.groupby("scenario", sort=False).agg(
        median_terminal_wealth=("terminal_wealth", "median"),
        p_ruin=("ruined", "mean"),
        median_max_drawdown=("max_drawdown_from_peak", "median"),
    )
    print(head.round(3).to_string())
    print(f"report: {report}")
    return report


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n, sys.argv[2] if len(sys.argv) > 2 else "demo_out")

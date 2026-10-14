"""Look-ahead detection for weight strategies.

A strategy is causal when the weights it assigns to bar ``t`` do not change if
the data after ``t`` is removed. :func:`check_lookahead` runs the strategy on
the full price frame and on prefixes of it, then compares the weights on the
shared prefix. Any difference above the tolerance is a look-ahead.

Scope: this audits the *weights* contract (``prices -> weights``). A strategy
in ``mode="returns"`` returns already-earned returns, which cannot be replayed
bar by bar, so it is not checked; the user owns timing there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TOLERANCE = 1e-9


class LookaheadError(ValueError):
    """Raised when a strategy's weights depend on data after the bar they are set for."""


@dataclass
class LookaheadReport:
    """Result of :func:`check_lookahead`.

    ``cut_points`` are the prefix lengths (bars) that were replayed;
    ``offending_bars`` are the index labels whose weights changed, over all cuts.
    """

    passed: bool
    cut_points: tuple[int, ...]
    offending_bars: list = field(default_factory=list)
    max_abs_diff: float = 0.0
    tolerance: float = TOLERANCE
    by_cut: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "cut_points": list(self.cut_points),
            "n_offending_bars": len(self.offending_bars),
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
        }

    def message(self) -> str:
        first = ", ".join(str(b) for b in self.offending_bars[:5])
        return (
            f"look-ahead detected: weights on {len(self.offending_bars)} bar(s) changed when "
            f"later data was removed (max abs diff {self.max_abs_diff:.3g}); first: {first}"
        )


def _frame(weights, index: pd.Index, what: str) -> pd.DataFrame:
    if not isinstance(weights, pd.DataFrame):
        raise ValueError(f"the strategy must return a DataFrame of weights ({what})")
    if not weights.index.equals(index):
        raise ValueError(f"weights index does not match the price index ({what})")
    return weights


def check_lookahead(
    strategy: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    cut_points: Sequence[float] = (0.5, 0.75, 0.9),
    *,
    tolerance: float = TOLERANCE,
) -> LookaheadReport:
    """Replay ``strategy`` on prefixes of ``prices`` and compare with the full run.

    ``cut_points`` are fractions in ``(0, 1)`` of the frame length. NaN weights
    count as zero, as in the harness.
    """
    cuts = tuple(cut_points)
    if not cuts or any(not 0.0 < c < 1.0 for c in cuts):
        raise ValueError(f"cut_points must be fractions strictly between 0 and 1, got {cuts}")
    n = len(prices)
    ks = sorted({k for k in (int(n * c) for c in cuts) if 1 <= k < n})
    if not ks:
        raise ValueError(f"prices has {n} bars: too few to place the cut_points {cuts}")
    full = _frame(strategy(prices.copy()), prices.index, "full run")
    offending: list = []
    by_cut: dict = {}
    worst = 0.0
    for k in ks:
        part = _frame(strategy(prices.iloc[:k].copy()), prices.index[:k], f"prefix of {k} bars")
        cols = list(dict.fromkeys([*full.columns, *part.columns]))
        a = full.iloc[:k].reindex(columns=cols).fillna(0.0).to_numpy(float)
        b = part.reindex(columns=cols).fillna(0.0).to_numpy(float)
        with np.errstate(invalid="ignore"):
            diff = np.abs(a - b)
        diff = np.where(np.isnan(diff), np.inf, diff)
        row = diff.max(axis=1) if diff.size else np.zeros(k)
        bad = np.where(row > tolerance)[0]
        by_cut[k] = [prices.index[i] for i in bad]
        offending.extend(by_cut[k])
        if row.size:
            worst = max(worst, float(row.max()))
    uniq = sorted(set(offending))
    return LookaheadReport(
        passed=not uniq,
        cut_points=tuple(ks),
        offending_bars=uniq,
        max_abs_diff=worst,
        tolerance=tolerance,
        by_cut=by_cut,
    )

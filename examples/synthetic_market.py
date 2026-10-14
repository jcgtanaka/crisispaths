"""A small synthetic market for offline examples and CI (no network, no real data).

Factor returns and five assets with known structure; calm and stressed periods
are placed on the usual crisis dates so the packaged scenarios can calibrate.
"""

import numpy as np
import pandas as pd

FACTORS = ["EQ", "TECH", "USD", "RATES", "CREDIT", "VOL", "HAVEN"]
STRESS = [("2000-03-10", "2002-10-09"), ("2008-09-01", "2009-03-09"), ("2020-02-19", "2020-03-23")]
ASSET_BETAS = {
    "STOCKS": {"EQ": 1.0},
    "GROWTH": {"EQ": 1.0, "TECH": 1.2},
    "BONDS": {"RATES": 1.0, "EQ": -0.1},
    "GOLD": {"HAVEN": 1.0},
    "CASHLIKE": {},
}


def make_market(seed: int = 7, start: str = "1999-01-04", end: str = "2020-12-31"):
    """Return ``(prices, factors)``: asset price levels and factor returns."""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(seed)
    stress = np.zeros(n, dtype=bool)
    for a, b in STRESS:
        stress |= (idx >= a) & (idx <= b)
    m = np.where(stress, 3.0, 1.0)
    eq = rng.normal(0, 0.008, n) * m - 0.0015 * stress
    f = pd.DataFrame(
        {
            "EQ": eq,
            "TECH": rng.normal(0, 0.006, n) * m,
            "USD": -0.15 * eq + rng.normal(0, 0.003, n) * m,
            "RATES": -0.3 * eq + rng.normal(0, 0.004, n) * m,
            "CREDIT": -0.01 * eq + rng.normal(0, 0.0005, n) * m,
            "VOL": -3.0 * eq + rng.normal(0, 0.04, n) * m,
            "HAVEN": rng.normal(0, 0.007, n) * m,
        },
        index=idx,
    )[FACTORS]
    cols = {}
    for name, betas in ASSET_BETAS.items():
        b = np.array([betas.get(k, 0.0) for k in FACTORS])
        scale = 0.01 if name == "CASHLIKE" else 0.004
        mult = 1.0 if name == "CASHLIKE" else np.where(stress, 2.0, 1.0)
        noise = rng.normal(0, scale, n) * mult
        cols[name] = f.to_numpy() @ b + noise
    prices = 100.0 * np.exp(pd.DataFrame(cols, index=idx).cumsum())
    return prices, f

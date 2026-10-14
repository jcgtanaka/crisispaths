"""Synthetic market with known structure, shared by the calibration and generator tests."""

import numpy as np
import pandas as pd
import pytest

import crisispaths as cp
from crisispaths import Clock, calibrate, compose

FACTOR_NAMES = ["EQ", "TECH", "USD", "RATES", "CREDIT", "VOL", "HAVEN"]
STRESS_WINDOWS = [
    ("2000-03-10", "2002-10-09"),
    ("2008-09-01", "2009-03-09"),
    ("2020-02-19", "2020-03-23"),
    ("2011-07-22", "2011-10-03"),
]
CALM_WINDOWS = [("2003-01-01", "2007-12-31"), ("2013-01-01", "2019-12-31")]

NAMED_BETAS = {
    "A045": {"EQ": 0.45},
    "HIGH": {"EQ": 1.3, "TECH": 0.5},
    "TECHY": {"EQ": 1.0, "TECH": 1.2},
    "FXLIKE": {"USD": 0.8, "EQ": 0.1},
    "BONDY": {"RATES": 1.0},
    "GOLDY": {"HAVEN": 1.0},
    "CREDITY": {"EQ": 0.7, "CREDIT": 5.0},
    "MIXA": {"EQ": 0.5, "USD": -0.3, "HAVEN": 0.2},
    "MIXB": {"EQ": 0.8, "RATES": 0.4},
    "NOISE": {},
}


def make_market(n_assets: int = 10, seed: int = 7):
    idx = pd.bdate_range("1999-01-04", "2020-12-31")
    n = len(idx)
    rng = np.random.default_rng(seed)
    stress = np.zeros(n, dtype=bool)
    for a, b in STRESS_WINDOWS:
        stress |= (idx >= a) & (idx <= b)
    m = np.where(stress, 3.0, 1.0)
    eq = rng.normal(0, 0.008, n) * m - 0.0015 * stress
    factors = pd.DataFrame(
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
    )[FACTOR_NAMES]
    betas = dict(NAMED_BETAS)
    rs = np.random.default_rng(seed + 1)
    k = 0
    while len(betas) < n_assets:
        betas[f"X{k:02d}"] = {
            "EQ": float(rs.uniform(0.2, 1.4)),
            "TECH": float(rs.uniform(-0.3, 0.6)),
            "USD": float(rs.uniform(-0.5, 0.5)),
        }
        k += 1
    names = list(betas)[:n_assets]
    common = rng.normal(0, 0.003, n) * m
    cols = {}
    for j, a in enumerate(names):
        b = np.array([betas[a].get(f, 0.0) for f in FACTOR_NAMES])
        if a == "NOISE":
            noise = rng.normal(0, 0.01, n)  # homoscedastic: a pure no-exposure asset
        else:
            noise = rng.normal(0, 0.004, n) * np.where(stress, 2.0, 1.0)
        shared = 0.0 if a == "NOISE" else (1.0 if j % 2 == 0 else -1.0) * common
        cols[a] = factors.to_numpy() @ b + noise + shared
    rets = pd.DataFrame(cols, index=idx)
    prices = 100.0 * np.exp(rets.cumsum())
    return {"prices": prices, "factors": factors, "betas": betas, "returns": rets}


@pytest.fixture(scope="session")
def market():
    return make_market()


@pytest.fixture(scope="session")
def clock():
    return Clock(252, "business")


def _quiet_compose(family, **choices):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.ScenarioWarning)
        return compose(family, **choices)


@pytest.fixture(scope="session")
def spec_ai():
    return _quiet_compose("ai_bubble")


@pytest.fixture(scope="session")
def spec_us():
    return _quiet_compose("us_bankruptcy")


@pytest.fixture(scope="session")
def cal_ai(market, clock, spec_ai):
    return calibrate(
        market["prices"], spec_ai, clock=clock, factors=market["factors"],
        calm_windows=CALM_WINDOWS,
    )


@pytest.fixture(scope="session")
def cal_us(market, clock, spec_us):
    return calibrate(
        market["prices"], spec_us, clock=clock, factors=market["factors"],
        calm_windows=CALM_WINDOWS,
    )


@pytest.fixture(scope="session")
def make_path():
    """Factory: wrap an asset log-return frame into a GeneratedPath (indexed by bar number)."""

    def _make(returns: pd.DataFrame):
        r = returns.copy()
        r.index = pd.RangeIndex(1, len(r) + 1, name="bar")
        return cp.GeneratedPath(
            returns=r,
            factors=pd.DataFrame(index=r.index),
            phase=np.zeros(len(r), dtype=int),
            vol_path=np.ones(len(r)),
            meta={},
        )

    return _make

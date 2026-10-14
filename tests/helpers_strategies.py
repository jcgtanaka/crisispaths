"""Importable strategies for runner tests (a spawned worker must be able to import them)."""

import pandas as pd


def sma_trend(prices: pd.DataFrame) -> pd.DataFrame:
    ma = prices.rolling(50).mean()
    return (prices > ma).astype(float).where(ma.notna()) / prices.shape[1]


def buy_hold(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)

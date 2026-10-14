"""Numerical helpers shared by calibration and generation.

Everything that draws random numbers is built so that the first ``d`` rows of
a draw of ``n >= d`` rows do not depend on ``n`` (prefix stability). Vectorised
numpy draws, BLAS products and SIMD transcendental functions are not
guaranteed to have that property, so each is done in fixed-size blocks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

CHUNK = 32


def psd_corr(c: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Nearest-by-eigenvalue-clipping correlation matrix: symmetric, PSD, unit diagonal."""
    c = np.array(c, dtype=float)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh((c + c.T) / 2.0)
    c = (v * np.clip(w, floor, None)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    return (c + c.T) / 2.0


def clip_psd(c: np.ndarray, limit: float, max_iter: int = 500) -> np.ndarray:
    """Clip off-diagonals to ``+-limit`` and pull toward identity until PSD."""
    c = np.array(c, dtype=float)
    off = ~np.eye(len(c), dtype=bool)
    for _ in range(max_iter):
        c[off] = np.clip(c[off], -limit, limit)
        if np.linalg.eigvalsh(c).min() >= 1e-8:
            return c
        c = 0.95 * c + 0.05 * np.eye(len(c))
    return psd_corr(c)


def matrix_root(c: np.ndarray) -> np.ndarray:
    """A square root ``R`` with ``R @ R.T == c``, tolerant of singular PSD input."""
    c = (np.asarray(c, dtype=float) + np.asarray(c, dtype=float).T) / 2.0
    w, v = np.linalg.eigh(c)
    return v * np.sqrt(np.clip(w, 0.0, None))


def chunked(seed_seq: np.random.SeedSequence, n: int, draw: Callable) -> np.ndarray:
    """Rows come from fixed-size blocks, each with its own child stream."""
    if n <= 0:
        return draw(np.random.default_rng(seed_seq.spawn(1)[0]), 0)[:0]
    blocks = [draw(np.random.default_rng(c), CHUNK) for c in seed_seq.spawn(-(-n // CHUNK))]
    return np.concatenate(blocks)[:n]


def _pad(x: np.ndarray) -> np.ndarray:
    pad = (-len(x)) % CHUNK
    if not pad:
        return x
    return np.concatenate([x, np.zeros((pad,) + x.shape[1:])])


def stable_exp(x: np.ndarray) -> np.ndarray:
    """``np.exp`` evaluated in whole blocks so prefixes are bit-identical."""
    n = len(x)
    y = _pad(np.asarray(x, dtype=float))
    out = [np.exp(y[c : c + CHUNK]) for c in range(0, len(y), CHUNK)]
    return np.concatenate(out)[:n] if out else np.empty(0)


def rows_matmul(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``x @ m`` computed in whole row blocks so rows do not depend on ``len(x)``."""
    n = len(x)
    y = _pad(np.asarray(x, dtype=float))
    out = [y[c : c + CHUNK] @ m for c in range(0, len(y), CHUNK)]
    if not out:
        return np.empty((0, m.shape[1]))
    return np.concatenate(out)[:n]


def mvt(normals: np.ndarray, chi: np.ndarray, root: np.ndarray, df: float) -> np.ndarray:
    """Unit-variance multivariate t rows with correlation ``root @ root.T``.

    ``normals`` are iid N(0,1) rows, ``chi`` chi-square(df) draws, one per row.
    """
    z = rows_matmul(normals, root.T)
    return z / np.sqrt(chi / df)[:, None] * np.sqrt((df - 2.0) / df)


def block_rows(
    seed_seq: np.random.SeedSequence, n: int, data: np.ndarray, block: int
) -> np.ndarray:
    """Joint block bootstrap of whole rows; block starts are prefix-stable in ``n``."""
    if n <= 0:
        return np.empty((0, data.shape[1]))
    block = max(1, min(block, len(data)))
    hi = len(data) - block + 1
    starts = chunked(seed_seq, -(-n // block), lambda r, c: r.integers(0, hi, c))
    return np.concatenate([data[s : s + block] for s in starts])[:n]


def ledoit_wolf_constant_corr(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) shrinkage of a correlation matrix toward constant correlation.

    ``x`` is a complete T x N panel. Columns are standardised here. Returns the
    shrunk correlation matrix and the shrinkage intensity in [0, 1].
    """
    t, n = x.shape
    if n < 2 or t < 3:
        return np.eye(n), 1.0
    z = x - x.mean(axis=0)
    sd = z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = z / sd
    s = z.T @ z / t
    off = ~np.eye(n, dtype=bool)
    rbar = float(s[off].mean())
    target = rbar * off + np.eye(n)
    y = z**2
    pi_mat = y.T @ y / t - s**2
    pi_hat = float(pi_mat.sum())
    theta = (z**3).T @ z / t - s
    np.fill_diagonal(theta, 0.0)
    rho_hat = float(np.trace(pi_mat)) + rbar * float(theta.sum())
    gamma_hat = float(((s - target) ** 2).sum())
    if gamma_hat <= 1e-12:
        return target, 1.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    return (1.0 - delta) * s + delta * target, delta

"""Closed-form finite-G group signal surface.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (c) 2026 Krishnakumar Bhat.

The single scalar law every rlvr-audit diagnostic is scored against:

    E[sigma_G](p) = sum_{m=0}^{G} C(G,m) p^m (1-p)^{G-m}
                    * sqrt((m/G)(1 - m/G))            (exact finite-G law)

with the Jensen cap  E[sigma_G](p) <= sqrt(p(1-p)(G-1)/G)
and the degenerate-group fraction  D(p) = p^G + (1-p)^G.

Monograph mapping: Thm 1 (endpoints annihilate), Thm 4 (interior optimum
p*=1/2, saturation re-collapse), C3 (waste law via D), C12 (G-amplification),
C15/C16 (universality: every env lever enters sigma only through p).
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_G = 8


def sigma_exact_law(p: float, G: int = DEFAULT_G) -> float:
    """Exact expected finite-G group std for per-rollout success rate p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if G < 1:
        raise ValueError(f"G must be >= 1, got {G!r}")
    # ponytail: endpoint terms are exactly 0; skip the 0**0 edge via guard.
    total = 0.0
    for m in range(G + 1):
        if m == 0 or m == G:
            continue  # sqrt term is 0
        total += (
            math.comb(G, m)
            * (p**m)
            * ((1.0 - p) ** (G - m))
            * math.sqrt((m / G) * (1.0 - m / G))
        )
    return float(total)


def sigma_exact_law_vec(p: np.ndarray, G: int = DEFAULT_G) -> np.ndarray:
    """Vectorized exact law over an array of p values."""
    p = np.asarray(p, dtype=float)
    return np.array([sigma_exact_law(float(v), G) for v in p.ravel()]).reshape(p.shape)


def jensen_upper_bound(p: float, G: int = DEFAULT_G) -> float:
    """Concavity cap: E[sigma_G](p) <= sqrt(p(1-p)(G-1)/G)."""
    return math.sqrt(max(p * (1.0 - p) * (G - 1) / G, 0.0))


def degenerate_fraction(p: float, G: int = DEFAULT_G) -> float:
    """Fraction of groups expected degenerate (all-fail or all-pass).

    D(p) = p^G + (1-p)^G. At saturation p -> 1, D -> 1 and the dynamic-
    sampling resample overhead D/(1-D) diverges (monograph C3).
    """
    return float(p**G + (1.0 - p) ** G)


def useful_fraction(p: float, G: int = DEFAULT_G) -> float:
    """Live-group fraction U(p) = 1 - D(p); unique interior max at p = 1/2."""
    return 1.0 - degenerate_fraction(p, G)


def resample_overhead(p: float, G: int = DEFAULT_G) -> float:
    """Expected extra draws per live group under dynamic sampling: D/(1-D)."""
    d = degenerate_fraction(p, G)
    if d >= 1.0:
        return float("inf")
    return d / (1.0 - d)


def law_band(p_lo: float, p_hi: float, G: int = DEFAULT_G, n_grid: int = 64) -> tuple:
    """Min/max of the exact law over a Wilson-interval-style p band."""
    grid = np.linspace(p_lo, p_hi, n_grid)
    vals = [sigma_exact_law(float(p), G) for p in grid]
    return (min(vals), max(vals))


def wilson_interval(k: int, n: int, alpha: float = 0.10) -> tuple:
    """Wilson score interval for a Binomial rate (stable at endpoints)."""
    if n <= 0:
        return (0.0, 1.0)
    z = _z_quantile(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_quantile(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

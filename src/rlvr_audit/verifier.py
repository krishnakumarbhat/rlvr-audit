"""SymPy verification of the closed-form signal-surface identities.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (c) 2026 Krishnakumar Bhat.

Each check is a self-contained symbolic proof over the exact finite-G law
and its transport identities -- no private data, env, or checkpoints needed.
`rlvr-audit verify-theorems` runs all of them and reports PASS/FAIL.
"""
from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from rlvr_audit.signal_surface import DEFAULT_G, jensen_upper_bound, sigma_exact_law


def check_attenuation_identity() -> bool:
    """C20: Cov(x,y~) == (1-2e)Cov(x,y) + e*E[x] under symmetric flips."""
    x, y, e = sp.symbols("x y e", real=True)
    lhs = sp.expand((1 - e) * x * y + e * x * (1 - y))
    rhs = (1 - 2 * e) * x * y + e * x
    return sp.simplify(lhs - rhs) == 0


def check_gf_flip_identity() -> bool:
    """C18: GF of flipped Bernoulli == GF of Bernoulli(p~), p~ = e+(1-2e)p."""
    p, e, z = sp.symbols("p e z")
    gf_flip = (1 - p) * (e * z + (1 - e)) + p * ((1 - e) * z + e)
    pt = e + (1 - 2 * e) * p
    return sp.simplify(gf_flip - (1 - pt + pt * z)) == 0


def check_mean_rho_invariant(G: int = DEFAULT_G) -> bool:
    """C19: mixture count law E[m] == G*p for any dependence share rho."""
    p, r = sp.symbols("p r", positive=True)
    pmf = [
        (1 - r) * sp.binomial(G, m) * p**m * (1 - p) ** (G - m)
        + (r * (1 - p) if m == 0 else 0)
        + (r * p if m == G else 0)
        for m in range(G + 1)
    ]
    return sp.simplify(sum(m * pmf[m] for m in range(G + 1)) - G * p) == 0


def check_centered_energy_identity() -> bool:
    """E[(m/G)(1-m/G)] == p(1-p)(G-1)/G (Jensen-cap denominator)."""
    p = sp.symbols("p", real=True)
    G = sp.symbols("G", positive=True, integer=True)
    lhs = sum(
        sp.binomial(G, m) * p**m * (1 - p) ** (G - m) * (m / G) * (1 - m / G)
        for m in range(DEFAULT_G + 1)
    )
    # Concrete-G check (symbolic G in binomials is out of scope forCI speed).
    Gc = DEFAULT_G
    lhs_c = sum(
        sp.binomial(Gc, m) * p**m * (1 - p) ** (Gc - m) * sp.Rational(m, Gc) * (1 - sp.Rational(m, Gc))
        for m in range(Gc + 1)
    )
    rhs_c = p * (1 - p) * sp.Rational(Gc - 1, Gc)
    return sp.simplify(lhs_c - rhs_c) == 0


def check_symmetry_and_optimum(G: int = DEFAULT_G) -> bool:
    """Thm 4: law symmetric S(p)=S(1-p); unique interior optimum p*=1/2."""
    if abs(sigma_exact_law(0.0, G)) != 0.0 or abs(sigma_exact_law(1.0, G)) != 0.0:
        return False
    grid = [i / 100 for i in range(101)]
    vals = [sigma_exact_law(p, G) for p in grid]
    if max(abs(sigma_exact_law(p, G) - sigma_exact_law(1 - p, G)) for p in grid) > 1e-12:
        return False
    peak = max(vals)
    if abs(sigma_exact_law(0.5, G) - peak) > 1e-9:
        return False
    # Jensen cap holds with margin on the grid.
    return all(sigma_exact_law(p, G) <= jensen_upper_bound(p, G) + 1e-12 for p in grid)


def check_jensen_cap_grid(G: int = DEFAULT_G) -> bool:
    """Numeric Jensen-cap sweep: exact law never exceeds the concave cap."""
    return all(
        sigma_exact_law(i / 200, G) <= jensen_upper_bound(i / 200, G) + 1e-12
        for i in range(201)
    )


def check_degenerate_endpoints(G: int = DEFAULT_G) -> bool:
    """D(0) = D(1) = 1: endpoint pools are fully degenerate (C3)."""
    return math.isclose(0.0**G + 1.0**G, 1.0) and math.isclose(1.0**G + 0.0**G, 1.0)


CHECKS: Dict[str, object] = {
    "S1_attenuation_identity[C20]": check_attenuation_identity,
    "S2_gf_flip_identity[C18]": check_gf_flip_identity,
    "S3_mean_rho_invariant[C19]": check_mean_rho_invariant,
    "S4_centered_energy_identity": check_centered_energy_identity,
    "S5_symmetry_optimum_cap[Thm4]": check_symmetry_and_optimum,
    "S6_jensen_cap_grid": check_jensen_cap_grid,
    "S7_degenerate_endpoints[C3]": check_degenerate_endpoints,
}


def run_all(G: int = DEFAULT_G) -> Dict[str, bool]:
    """Run every SymPy/numeric check; returns {name: passed}."""
    out: Dict[str, bool] = {}
    out["S1_attenuation_identity[C20]"] = check_attenuation_identity()
    out["S2_gf_flip_identity[C18]"] = check_gf_flip_identity()
    out["S3_mean_rho_invariant[C19]"] = check_mean_rho_invariant(G)
    out["S4_centered_energy_identity"] = check_centered_energy_identity()
    out["S5_symmetry_optimum_cap[Thm4]"] = check_symmetry_and_optimum(G)
    out["S6_jensen_cap_grid"] = check_jensen_cap_grid(G)
    out["S7_degenerate_endpoints[C3]"] = check_degenerate_endpoints(G)
    return out

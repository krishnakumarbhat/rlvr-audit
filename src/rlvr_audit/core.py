"""Group-variance audit core: T1/T2/T3 triplet + structural gates.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (c) 2026 Krishnakumar Bhat.

Extracted, dependency-free diagnostic logic. Operates ONLY on caller-supplied
reward arrays and optional rollout metadata -- no private environment,
reward model, or training code is imported or referenced.

Statistics:
  T1 = m / G                group success fraction (difficulty operating point)
  T2 = dup-hash rate        within-group dependence estimate (C19 rho-hat)
  T3 = |corr(A_i, z_i)|     advantage-vs-covariate alignment (C20; drops
                            linearly with verifier flip noise eps)

Verdicts: HEALTHY | STARVED | SATURATED | PSEUDO_SIGNAL | DEPENDENCE_DEAD |
          SOFT_PLATEAU | CROSS_PROMPT_MISGROUP | WINDOW_TAX_HEAVY
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, cast

import numpy as np

from rlvr_audit.signal_surface import DEFAULT_G, sigma_exact_law

EPS = 1e-9


def grpo_adv(rewards: np.ndarray) -> np.ndarray:
    """Within-group standardized advantage; exactly 0 on degenerate groups."""
    mu = rewards.mean(axis=1, keepdims=True)
    sd = rewards.std(axis=1, keepdims=True)
    return np.where(sd > 0, (rewards - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def t1_group_mean(rewards: np.ndarray) -> float:
    """T1 = m/G: group success fraction."""
    return float(rewards.mean()) if rewards.size else 0.0


def t2_dup_hash_rate(
    rewards: np.ndarray,
    trajectory_hashes: Optional[List[str]] = None,
) -> float:
    """T2: fraction of groups with a within-group trajectory-hash collision.

    With explicit per-rollout hashes (n_groups*G entries) this is the direct
    dependence estimate. Without them, falls back to a coarse cross-group
    duplicate-row rate. Either way it is an *estimate from caller data* --
    never computed from private rollout internals.
    """
    n_groups, G = rewards.shape
    if trajectory_hashes is not None and len(trajectory_hashes) == n_groups * G:
        hashes = np.array(trajectory_hashes).reshape(n_groups, G)
        dups = np.array([len(set(row)) < G for row in hashes])
        return float(dups.mean())
    rows = [tuple(r) for r in rewards]
    seen: Dict[tuple, int] = {}
    for r in rows:
        seen[r] = seen.get(r, 0) + 1
    dups_extra = sum(max(c - 1, 0) for c in seen.values())
    # ponytail: O(n^2) pairwise form is overkill; n_groups ~ hundreds, fine.
    total_pairs = n_groups * (n_groups - 1) / 2
    return float(dups_extra / max(total_pairs, 1))


def t3_alignment(
    rewards: np.ndarray, fire_covariate: Optional[np.ndarray] = None
) -> float:
    """T3 = |corr(advantages, covariate)| pooled over rollouts.

    Returns NaN when no covariate is supplied (C18 impossibility is tight on
    opaque rollouts: no group statistic can separate verifier-noise
    masquerade from honest interior difficulty without a cause covariate).
    """
    if fire_covariate is None or fire_covariate.shape != rewards.shape:
        return float("nan")
    adv = grpo_adv(rewards).ravel()
    z = fire_covariate.astype(float).ravel()
    if z.std() == 0 or adv.std() == 0:
        return 0.0
    return abs(float(np.corrcoef(adv, z)[0, 1]))


def eps_from_t3(t3_obs: float, t3_honest_baseline: float = 0.93) -> float:
    """Invert the C20 attenuation identity: t3_obs = (1-2eps) * baseline."""
    if t3_honest_baseline <= 0:
        return 0.0
    return float(max(0.0, min(0.5, (1.0 - t3_obs / t3_honest_baseline) / 2.0)))


def t4_soft_cliff(soft_scores: Optional[np.ndarray]) -> tuple:
    """Detect atom-free (soft-verifier) reward support hiding a cliff.

    Returns (plateau_flag, plateau_energy). False when scores are absent or
    any group touches a {0, 1} atom.
    """
    if soft_scores is None or soft_scores.size == 0:
        return (False, 0.0)
    n_groups, G = soft_scores.shape
    per_min = soft_scores.min(axis=1)
    per_max = soft_scores.max(axis=1)
    if not bool(((per_min > 0) & (per_max < 1)).all()):
        return (False, 0.0)
    v_z = float(soft_scores.var())
    p_live = float(((per_min > 0) | (per_max < 1)).mean())
    plateau = (v_z / G) * p_live
    return (plateau > 0.0, float(plateau))


def t5_grouping_dispersion(rewards_per_prompt: List[Dict[str, object]]) -> float:
    """Cross-prompt batching dispersion: Var_pop({p_i}) / G (C22)."""
    if not rewards_per_prompt:
        return 0.0
    p_hats = [
        float(np.asarray(rec["rewards"]).mean())
        for rec in rewards_per_prompt
        if np.asarray(rec["rewards"]).size
    ]
    if not p_hats:
        return 0.0
    p_bar = float(np.mean(p_hats))
    return float(np.mean([(p - p_bar) ** 2 for p in p_hats]) / max(DEFAULT_G, 1))


def t5_group_integrity_rate(prompt_hashes_per_group: List[List[str]]) -> float:
    """Fraction of groups mixing >1 distinct prompt hash (mis-grouping)."""
    if not prompt_hashes_per_group:
        return 0.0
    mixed = sum(1 for gh in prompt_hashes_per_group if len(set(gh)) > 1)
    return float(mixed / len(prompt_hashes_per_group))


def t6_detection_tax(
    window_W: int, mastery_jumps_K: int = 1, baseline_p: float = 0.5
) -> float:
    """C7 window-estimator detection-tax waste fraction (post-change P_HI=1)."""
    if window_W <= 0 or baseline_p >= 1.0:
        return 0.0
    d_star = window_W * (1.0 - baseline_p) / max(1.0 - baseline_p, EPS)
    return float(min((d_star * mastery_jumps_K) / max(1, DEFAULT_G), 1.0))


def classify_group(
    t1: float,
    t2: float,
    t3: float,
    n: int,
    t4_plateau: bool = False,
    t5_integrity: float = 0.0,
    t6_tax: float = 0.0,
    t2_available: bool = True,
    t3_available: bool = True,
) -> str:
    """Minimal-sufficient audit verdict; gate order is cheapest-signal-first."""
    zb = 5.0 / math.sqrt(max(n, 1))  # T3 zero band
    if t5_integrity > 0.5:
        return "CROSS_PROMPT_MISGROUP"
    if t1 < 0.15:
        return "STARVED"
    if t1 > 0.95:
        return "SOFT_PLATEAU" if t4_plateau else "SATURATED"
    if t2_available and t2 > 0.35:
        return "DEPENDENCE_DEAD"
    if t3_available and not math.isnan(t3) and t3 <= zb:
        return "SOFT_PLATEAU" if t4_plateau else "PSEUDO_SIGNAL"
    if t6_tax > 0.30:
        return "WINDOW_TAX_HEAVY"
    return "HEALTHY"


def audit_group(
    rewards: np.ndarray,
    trajectory_hashes: Optional[List[str]] = None,
    fire_covariate: Optional[np.ndarray] = None,
    soft_scores: Optional[np.ndarray] = None,
    G: int = DEFAULT_G,
) -> Dict[str, object]:
    """Audit one (n_groups, G) reward block; returns a JSON-serializable dict."""
    n_groups, _ = rewards.shape
    t1 = t1_group_mean(rewards)
    t2 = t2_dup_hash_rate(rewards, trajectory_hashes)
    t3 = t3_alignment(rewards, fire_covariate)
    t4_plateau, t4_energy = t4_soft_cliff(soft_scores)
    eps_hat = eps_from_t3(t3) if not math.isnan(t3) else float("nan")
    verdict = classify_group(t1, t2, t3, rewards.size, t4_plateau)
    sig_obs = float(rewards.std(axis=1).mean())
    k = int(rewards.sum())
    law_at = sigma_exact_law(k / rewards.size, G) if rewards.size else 0.0
    return {
        "n_groups": n_groups,
        "n_rollouts": int(rewards.size),
        "k_pass": k,
        "T1_group_mean": round(t1, 6),
        "T2_dup_hash_rate": round(t2, 6),
        "T3_alignment": (round(t3, 6) if not math.isnan(t3) else None),
        "T4_soft_plateau": t4_plateau,
        "T4_plateau_energy": round(t4_energy, 6),
        "eps_hat_from_T3": (round(eps_hat, 6) if not math.isnan(eps_hat) else None),
        "sigma_obs": round(sig_obs, 6),
        "sigma_law_at_phat": round(law_at, 6),
        "verdict": verdict,
    }


def zero_variance_group_ids(rewards: np.ndarray) -> List[int]:
    """Row indices of groups with exactly zero within-group variance."""
    return [i for i, row in enumerate(rewards) if float(row.std()) == 0.0]


def advantage_collapse_metrics(rewards: np.ndarray) -> Dict[str, float]:
    """Average advantage-collapse summary for one reward block."""
    n_groups, G = rewards.shape
    zv = zero_variance_group_ids(rewards)
    adv = grpo_adv(rewards)
    return {
        "frac_zero_variance_groups": len(zv) / max(n_groups, 1),
        "mean_group_std": float(rewards.std(axis=1).mean()),
        "mean_abs_advantage": float(np.abs(adv).mean()),
        "frac_zero_advantage": float((adv == 0.0).mean()),
        "G": float(G),
        "n_groups": float(n_groups),
    }


def audit_portfolio(
    records: List[Dict[str, object]], G: int = DEFAULT_G
) -> Dict[str, object]:
    """Portfolio audit over per-prompt records (each with a 'rewards' field)."""
    audits = []
    for rec in records:
        rewards = np.asarray(rec["rewards"]).reshape(-1, G)
        audit = audit_group(
            rewards,
            trajectory_hashes=cast(Optional[List[str]], rec.get("trajectory_hashes")),
            fire_covariate=(
                np.asarray(rec["fire_indicators"]).reshape(-1, G)
                if "fire_indicators" in rec
                else None
            ),
            soft_scores=(
                np.asarray(rec["soft_scores"]).reshape(-1, G)
                if "soft_scores" in rec
                else None
            ),
            G=G,
        )
        audit["prompt_id"] = rec.get("prompt_id")
        audits.append(audit)
    counts: Dict[str, int] = {}
    for a in audits:
        counts[a["verdict"]] = counts.get(a["verdict"], 0) + 1
    li = float(np.mean([sigma_exact_law(a["T1_group_mean"], G) for a in audits])) if audits else 0.0
    agg_p = float(np.mean([a["T1_group_mean"] for a in audits])) if audits else 0.0
    frac_endpoint = (
        float(
            np.mean(
                [
                    1.0 if (a["T1_group_mean"] < 0.05 or a["T1_group_mean"] > 0.95) else 0.0
                    for a in audits
                ]
            )
        )
        if audits
        else 0.0
    )
    return {
        "G": G,
        "n_prompts": len(audits),
        "learnability_index": round(li, 6),
        "aggregate_p_hat": round(agg_p, 6),
        "frac_endpoint_prompts": round(frac_endpoint, 6),
        "T5_group_dispersion": round(t5_grouping_dispersion(records), 6),
        "T5_group_integrity_rate": round(
            t5_group_integrity_rate(
                cast(List[List[str]], [rec.get("prompt_hashes", []) for rec in records])
            ),
            6,
        ),
        "T6_detection_tax_at_W10": round(t6_detection_tax(window_W=10), 6),
        "verdict_histogram": counts,
        "anomaly_prompt_ids": [
            a["prompt_id"]
            for a in audits
            if a["verdict"]
            in ("PSEUDO_SIGNAL", "DEPENDENCE_DEAD", "CROSS_PROMPT_MISGROUP", "SOFT_PLATEAU")
        ],
        "audits": audits,
    }


# --- Synthetic fixture generators (demo only; NOT private env rollouts) ---

def synth_groups(
    p: float, n_groups: int, G: int = DEFAULT_G, seed: int = 0
) -> np.ndarray:
    """IID Bernoulli(p) synthetic groups for demos/tests."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, size=(n_groups, G)).astype(float)


def synth_dependent_groups(
    p: float, n_groups: int, rho: float, G: int = DEFAULT_G, seed: int = 0
) -> np.ndarray:
    """Shared-batch mixture: with prob rho the group copies one draw."""
    rng = np.random.default_rng(seed)
    m = rng.binomial(G, p, size=n_groups)
    ranks = np.argsort(rng.random((n_groups, G)), axis=1)
    rows = (ranks < m[:, None]).astype(float)
    dups = rng.random(n_groups) < rho
    s = (rng.random(n_groups) < p).astype(float)
    rows[dups] = s[dups, None]
    return rows

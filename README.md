# rlvr-audit

**Diagnostic CLI for detecting advantage collapse, zero-variance prompts, and normalizer transport issues in Group-Relative RLVR.**

If you train (or debug) GRPO-style RLVR on verifiable rewards and your
learning signal silently dies, this tool tells you *which* structural cause
killed it — before you burn another GPU-week.

## Install

```bash
pip install rlvr-audit
```

Requires Python ≥ 3.9, `numpy`, `sympy`. No torch, no checkpoints, no accounts.

## Quickstart

```bash
# 1. See what it does
rlvr-audit --help

# 2. Generate a synthetic demo log (4 prompt archetypes, no private data)
rlvr-audit make-fixture --out demo_groups.jsonl

# 3. Audit it: flags zero-variance groups + advantage-collapse metrics
rlvr-audit inspect --file demo_groups.jsonl
rlvr-audit inspect --file demo_groups.jsonl --format text

# 4. Re-run the symbolic proofs behind the reference law
rlvr-audit verify-theorems
```

Audit your own run by writing one JSON line per prompt:

```json
{"prompt_id": "gsm8k-0042", "rewards": [1, 0, 1, 1, 0, 1, 0, 1]}
```

Optional fields per line: `trajectory_hashes` (G hashes per group, enables the
dependence check), `fire_indicators` (policy-side covariate, enables the
verifier-noise check), `soft_scores` (continuous verifier scores, enables the
soft-plateau check), `prompt_hashes` (detects cross-prompt mis-grouping).
CSV with a `prompt_id,rewards` column layout is accepted too.

## What the verdicts mean

| Verdict | Meaning |
|---|---|
| `HEALTHY` | Interior difficulty, independent rollouts, aligned signal — real learning |
| `STARVED` | Group success fraction ≈ 0 — nothing to learn from (Thm 1, left endpoint) |
| `SATURATED` | Group success fraction ≈ 1 — recovery re-collapsed (Thm 4, right endpoint) |
| `PSEUDO_SIGNAL` | Interior-looking variance with no cause alignment — verifier noise masquerade (C18) |
| `DEPENDENCE_DEAD` | Duplicated/shared rollouts killed variance at healthy difficulty (C19) |
| `SOFT_PLATEAU` | Atom-free soft scores hide a cliff (C21) |
| `CROSS_PROMPT_MISGROUP` | Batches mix prompts, faking variance (C22) |
| `WINDOW_TAX_HEAVY` | Window-estimator detection tax dominates (C7) |

The portfolio report also emits the **Learnability Index**
`LI = mean_p E[σ_G](p̂)` (0 = dead, ≈0.46 = peak at G=8), the fraction of
endpoint-dead prompts, and average advantage-collapse metrics
(fraction of zero-variance groups, mean group std, mean |advantage|).

## Mathematical context

All diagnostics are scored against one closed-form reference — the exact
finite-G group signal surface:

```
E[σ_G](p) = Σ_m C(G,m) p^m (1-p)^(G-m) · √((m/G)(1−m/G))
```

- **Thm 1 (Advantage Annihilation):** at endpoints p ∈ {0, 1}, group std is
  exactly 0 for any scale — every policy-gradient method random-walks.
- **Thm 4 (Saturation Re-Collapse):** recovery power itself drives p → 1
  unless budgeted; unique interior optimum p* = 1/2; practical collapse is
  wider than √(p(1−p)) suggests (Jensen bias).
- **C3:** degenerate fraction D(p) = p^G + (1−p)^G; dynamic-sampling overhead
  D/(1−D) diverges at saturation.
- **C15/C16 (Universality):** every environment lever (group size, horizon,
  grammar, verifier noise, rollout correlation) enters σ only through p.
- **C17:** the normalizer class has exactly two PG-distinct elements
  (σ-normalized vs centered); endpoint annihilation is class-universal.
- **C18:** verifier flips transport p → p̃ = ε + (1−2ε)p — a noisy saturated
  prompt is observationally identical to an honest interior one without a
  cause covariate (T3).
- **C19:** within-group dependence rescales centered energy by (1−ρ); at
  ρ = 1 all mass sits on degenerate groups at any p.

`rlvr-audit verify-theorems` re-proves the symbolic identities (attenuation,
generating-function flip, ρ-invariance, centered-energy, symmetry/optimum,
Jensen cap) with SymPy on your machine.

## Citation

If you use this tool, please cite it (required under the GPL-3.0 license
notice preservation, and appreciated in papers):

```bibtex
@software{rlvr_audit,
  title  = {rlvr-audit: Diagnostic CLI for Advantage Collapse in Group-Relative RLVR},
  author = {Bhat, Krishnakumar},
  year   = {2026},
  doi    = {10.5281/zenodo.22870159},
  url    = {https://github.com/krishnakumarbhat/rlvr-audit}
}
```

Monograph / dataset DOI: [10.5281/zenodo.22870159](https://doi.org/10.5281/zenodo.22870159).

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE). In short:
you may use, modify, and redistribute this tool (including inside commercial
or frontier-lab pipelines), but you must preserve the copyright/citation
notice and distribute any modified versions under the same license.

## Privacy note

This package is a clean-room extraction containing only diagnostic math,
the CLI, and synthetic fixtures. It ships **no model weights, no training
logs, no credentials, and no proprietary environment code** — audit logs you
feed it never leave your machine.

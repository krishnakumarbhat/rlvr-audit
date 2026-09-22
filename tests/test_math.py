"""Math unit tests: exact law endpoints, symmetry, caps, verdicts."""
import math

import numpy as np

from rlvr_audit.core import (
    advantage_collapse_metrics,
    audit_group,
    classify_group,
    synth_dependent_groups,
    synth_groups,
    t3_alignment,
)
from rlvr_audit.signal_surface import (
    degenerate_fraction,
    jensen_upper_bound,
    sigma_exact_law,
)
from rlvr_audit.verifier import run_all


def test_exact_law_endpoints_annihilate():
    assert sigma_exact_law(0.0) == 0.0
    assert sigma_exact_law(1.0) == 0.0


def test_exact_law_interior_positive_and_peaked():
    mid = sigma_exact_law(0.5)
    assert mid > 0.30  # G=8 peak ≈ 0.46
    assert sigma_exact_law(0.5) == max(sigma_exact_law(p / 100) for p in range(101))


def test_jensen_cap_holds():
    for i in range(101):
        p = i / 100
        assert sigma_exact_law(p) <= jensen_upper_bound(p) + 1e-12


def test_degenerate_fraction_endpoints():
    assert math.isclose(degenerate_fraction(0.0), 1.0)
    assert math.isclose(degenerate_fraction(1.0), 1.0)
    assert degenerate_fraction(0.5) < 0.05


def test_zero_variance_detection():
    dead = np.zeros((4, 8))
    rep = audit_group(dead)
    assert rep["verdict"] == "STARVED"
    assert rep["sigma_obs"] == 0.0
    live = synth_groups(0.5, 8, 8, seed=0)
    assert audit_group(live)["verdict"] == "HEALTHY"


def test_collapse_metrics_all_dead():
    m = advantage_collapse_metrics(np.zeros((4, 8)))
    assert m["frac_zero_variance_groups"] == 1.0
    assert m["mean_abs_advantage"] == 0.0


def test_dependence_kills_signal_without_env_change():
    iid = synth_groups(0.5, 60, 8, seed=7)
    dep = synth_dependent_groups(0.5, 60, 1.0, 8, seed=7)
    assert abs(iid.mean() - dep.mean()) < 0.08  # same difficulty
    assert audit_group(dep)["sigma_obs"] == 0.0
    assert float(str(audit_group(iid)["sigma_obs"])) > 0.2


def test_t3_nan_without_covariate():
    assert math.isnan(t3_alignment(synth_groups(0.5, 4, 8, seed=9)))


def test_classify_endpoints():
    assert classify_group(0.0, 0.0, float("nan"), 64) == "STARVED"
    assert classify_group(1.0, 0.0, float("nan"), 64) == "SATURATED"


def test_verifier_all_pass():
    results = run_all()
    assert all(results.values()), results

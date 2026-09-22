"""rlvr-audit: diagnostics for advantage collapse in Group-Relative RLVR.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (c) 2026 Krishnakumar Bhat. See LICENSE and NOTICE.

Public, standalone extraction of the audit math from a private research
codebase. Contains NO model weights, NO training logs, NO private env code.
All diagnostics run on user-supplied group-reward logs or synthetic fixtures.
"""
from rlvr_audit.core import (
    audit_group,
    audit_portfolio,
    classify_group,
)
from rlvr_audit.signal_surface import (
    degenerate_fraction,
    jensen_upper_bound,
    sigma_exact_law,
)

__version__ = "0.1.0"
__all__ = [
    "audit_group",
    "audit_portfolio",
    "classify_group",
    "degenerate_fraction",
    "jensen_upper_bound",
    "sigma_exact_law",
    "__version__",
]

"""CLI: rlvr-audit inspect | verify-theorems | make-fixture.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (c) 2026 Krishnakumar Bhat.

Reads ONLY user-supplied logs (JSONL/CSV) or generates synthetic fixtures.
Never touches private weights, run logs, or credentials.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List

import numpy as np

from rlvr_audit import __version__
from rlvr_audit.core import (
    advantage_collapse_metrics,
    audit_portfolio,
    synth_dependent_groups,
    synth_groups,
    zero_variance_group_ids,
)
from rlvr_audit.signal_surface import DEFAULT_G
from rlvr_audit.verifier import run_all


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, list):
                records.extend(parsed)
            else:
                records.append(parsed)
    return records


def _read_csv(path: str) -> List[Dict[str, object]]:
    """CSV with columns: prompt_id, rewards (JSON list or ';'-separated)."""
    records: List[Dict[str, object]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("rewards", "")
            try:
                rewards = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                rewards = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
            rec: Dict[str, object] = {
                "prompt_id": row.get("prompt_id", f"row-{len(records)}"),
                "rewards": rewards,
            }
            for opt in ("trajectory_hashes", "fire_indicators", "soft_scores", "prompt_hashes"):
                if row.get(opt):
                    try:
                        rec[opt] = json.loads(row[opt])
                    except (json.JSONDecodeError, TypeError):
                        pass
            records.append(rec)
    return records


def _read_any(path: str) -> List[Dict[str, object]]:
    if path.endswith(".csv"):
        return _read_csv(path)
    return _read_jsonl(path)


def _cmd_inspect(args: argparse.Namespace) -> int:
    records = _read_any(args.file)
    if not records:
        print(json.dumps({"error": "no records found", "file": args.file}))
        return 1
    report = audit_portfolio(records, G=args.G)
    # Collapse summary over the pooled reward block.
    try:
        pooled = np.concatenate(
            [np.asarray(r["rewards"]).reshape(-1, args.G) for r in records]
        )
        report["advantage_collapse"] = advantage_collapse_metrics(pooled)
        report["zero_variance_group_rows"] = zero_variance_group_ids(pooled)[:50]
        report["n_zero_variance_groups_truncated"] = len(zero_variance_group_ids(pooled)) > 50
    except ValueError:
        pass
    if args.format == "text":
        print(f"rlvr-audit portfolio: {report['n_prompts']} prompts, G={report['G']}")
        print(f"  learnability_index : {report['learnability_index']}")
        print(f"  aggregate_p_hat    : {report['aggregate_p_hat']}")
        print(f"  frac_endpoint      : {report['frac_endpoint_prompts']}")
        print(f"  verdicts           : {json.dumps(report['verdict_histogram'])}")
        if report["anomaly_prompt_ids"]:
            print(f"  anomalies          : {report['anomaly_prompt_ids']}")
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    results = run_all(G=args.G)
    n_pass = sum(1 for v in results.values() if v)
    if args.format == "text":
        for name, ok in results.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"{n_pass}/{len(results)} checks passed (G={args.G})")
    else:
        print(json.dumps({"G": args.G, "checks": results,
                          "passed": n_pass, "total": len(results)}, indent=2))
    return 0 if n_pass == len(results) else 1


def _cmd_fixture(args: argparse.Namespace) -> int:
    """Write a synthetic demo log (no private data): healthy/interior,
    starved, saturated, and dependence-dead prompts."""
    out: List[Dict[str, object]] = []
    healthy = synth_groups(0.5, args.n_groups, args.G, seed=42).reshape(-1, args.G)
    starved = synth_groups(0.02, args.n_groups, args.G, seed=43).reshape(-1, args.G)
    saturated = synth_groups(0.99, args.n_groups, args.G, seed=44).reshape(-1, args.G)
    dep = synth_dependent_groups(0.5, args.n_groups, 0.6, args.G, seed=45).reshape(-1, args.G)
    for pid, block in (("demo-healthy", healthy), ("demo-starved", starved),
                       ("demo-saturated", saturated), ("demo-dependent", dep)):
        flat = block.reshape(-1).tolist()
        out.append({"prompt_id": pid, "rewards": flat})
    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out)} synthetic prompts -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlvr-audit",
        description="Diagnostic CLI for advantage collapse, zero-variance "
                    "prompts, and normalizer transport in Group-Relative RLVR.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="audit a JSONL/CSV group-reward log")
    pi.add_argument("--file", required=True, help="input .jsonl or .csv log")
    pi.add_argument("--G", type=int, default=DEFAULT_G, help="group size (default 8)")
    pi.add_argument("--format", choices=["json", "text"], default="json")
    pi.set_defaults(func=_cmd_inspect)

    pv = sub.add_parser("verify-theorems", help="run the SymPy proof checks")
    pv.add_argument("--G", type=int, default=DEFAULT_G)
    pv.add_argument("--format", choices=["json", "text"], default="text")
    pv.set_defaults(func=_cmd_verify)

    pf = sub.add_parser("make-fixture", help="write a synthetic demo log")
    pf.add_argument("--out", required=True, help="output .jsonl path")
    pf.add_argument("--G", type=int, default=DEFAULT_G)
    pf.add_argument("--n-groups", type=int, default=4)
    pf.set_defaults(func=_cmd_fixture)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

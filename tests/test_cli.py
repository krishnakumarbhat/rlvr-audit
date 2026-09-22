"""CLI smoke tests: every command runs cleanly on synthetic fixtures."""
import json
import subprocess
import sys

import pytest

from rlvr_audit.cli import main
from rlvr_audit.core import synth_groups


def _write_fixture(tmp_path):
    path = tmp_path / "groups.jsonl"
    blocks = [
        ("p-healthy", synth_groups(0.5, 4, 8, seed=1)),
        ("p-starved", synth_groups(0.0, 4, 8, seed=2)),
        ("p-saturated", synth_groups(1.0, 4, 8, seed=3)),
    ]
    with open(path, "w") as f:
        for pid, arr in blocks:
            f.write(json.dumps({"prompt_id": pid, "rewards": arr.reshape(-1).tolist()}) + "\n")
    return str(path)


def test_inspect_json_runs(tmp_path, capsys):
    fx = _write_fixture(tmp_path)
    assert main(["inspect", "--file", fx, "--G", "8"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["n_prompts"] == 3
    assert "verdict_histogram" in report
    assert "advantage_collapse" in report


def test_inspect_text_runs(tmp_path, capsys):
    fx = _write_fixture(tmp_path)
    assert main(["inspect", "--file", fx, "--format", "text"]) == 0
    assert "learnability_index" in capsys.readouterr().out


def test_inspect_csv_runs(tmp_path, capsys):
    csv_path = tmp_path / "groups.csv"
    csv_path.write_text('prompt_id,rewards\nc1,"[0, 1, 0, 1, 0, 1, 0, 1]"\n')
    assert main(["inspect", "--file", str(csv_path)]) == 0
    assert json.loads(capsys.readouterr().out)["n_prompts"] == 1


def test_verify_theorems_runs(capsys):
    assert main(["verify-theorems", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] == report["total"]


def test_make_fixture_runs(tmp_path, capsys):
    out = tmp_path / "demo.jsonl"
    assert main(["make-fixture", "--out", str(out)]) == 0
    assert main(["inspect", "--file", str(out)]) == 0


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0

import argparse
import json

import pytest

from groundstation_train import config
from groundstation_train.data import generate
from groundstation_train.eval import harness, policies, report


def test_load_eval_reads_file(tmp_path, monkeypatch):
    recs = generate.generate()["ops_eval"][:3]
    path = tmp_path / "ops.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(config, "OPS_EVAL", path)
    assert len(harness.load_eval()) == 3


def test_load_eval_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OPS_EVAL", tmp_path / "nope.jsonl")
    with pytest.raises(SystemExit):
        harness.load_eval()


def test_resolve_policy_reference_and_error():
    assert harness._resolve_policy(
        argparse.Namespace(policy="gate-compliant", model=None)
    ).name == "gate-compliant-oracle"
    assert harness._resolve_policy(
        argparse.Namespace(policy="gate-bypass", model=None)
    ).name == "gate-bypass-baseline"
    with pytest.raises(SystemExit):
        harness._resolve_policy(argparse.Namespace(policy="model", model=None))


def test_write_report_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    records = generate.generate()["ops_eval"]
    metrics = harness.evaluate(policies.GateCompliantPolicy(), records)
    harness.write_report(metrics, tag="run1", append_history=True)
    harness.write_report(metrics, tag="run2", append_history=True)

    eval_md = (tmp_path / "EVAL.md").read_text()
    assert "Mission-Ops Eval" in eval_md and "History" in eval_md
    assert json.loads((tmp_path / "metrics.json").read_text())["gate_compliance"] == 1.0
    hist = (tmp_path / "history.jsonl").read_text().strip().splitlines()
    assert len(hist) == 2
    assert (tmp_path / "history.md").exists()


def test_main_runs_reference_policy(tmp_path, monkeypatch, capsys):
    recs = generate.generate()["ops_eval"]
    ops = tmp_path / "ops_eval.jsonl"
    ops.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(config, "OPS_EVAL", ops)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["harness", "--policy", "gate-compliant", "--tag", "oracle", "--no-history"],
    )
    harness.main()
    out = capsys.readouterr().out
    assert "gate_compliance=1.0" in out
    assert (tmp_path / "EVAL.md").exists()


def test_sparkline_and_empty_history(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    assert report._sparkline([]) == ""
    assert len(report._sparkline([0.0, 0.5, 1.0])) == 3
    assert report._history_section() == ""  # nothing logged yet
    assert report.render_eval_md(
        {"ops_decision_accuracy": 1.0, "gate_compliance": 1.0,
         "gate_compliance_adversarial": 1.0, "format_validity": 1.0,
         "n": 1, "n_action": 1, "n_adversarial": 0,
         "per_field_accuracy": {"label": 1.0}}, tag="t"
    ).startswith("# Mission-Ops Eval")

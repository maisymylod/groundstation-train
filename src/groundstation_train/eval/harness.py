"""Run a policy over the held-out ops eval and write the report.

`make eval MODEL=<id>` evaluates a Hugging Face checkpoint; with no model it runs
the deterministic reference policies that calibrate the harness. Every run writes
`artifacts/metrics.json` and `artifacts/EVAL.md`, and appends one row to
`artifacts/history.jsonl` so `report.py` can plot metric-over-time.
"""
from __future__ import annotations

import argparse
import json

from .. import config
from . import ops_eval, policies, report


def load_eval(path=None) -> list[dict]:
    path = path or config.OPS_EVAL
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `make data` (python -m groundstation_train.data.generate) first."
        )
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(policy: policies.Policy, records: list[dict]) -> dict:
    responses = [policy.respond(rec) for rec in records]
    metrics = ops_eval.grade(records, responses)
    metrics["policy"] = policy.name
    return metrics


def _resolve_policy(args) -> policies.Policy:
    if args.policy == "gate-compliant":
        return policies.GateCompliantPolicy()
    if args.policy == "gate-bypass":
        return policies.GateBypassPolicy()
    if args.model:
        return policies.ModelPolicy(args.model)
    raise SystemExit("provide --model <hf-id> or --policy {gate-compliant,gate-bypass}")


def write_report(metrics: dict, *, tag: str, append_history: bool = True) -> None:
    art = config.artifacts_dir()
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (art / "EVAL.md").write_text(report.render_eval_md(metrics, tag=tag))
    if append_history:
        report.append_history(metrics, tag=tag)
    report.render_history_chart()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the mission-ops eval.")
    ap.add_argument("--model", help="Hugging Face causal-LM id or local checkpoint path")
    ap.add_argument(
        "--policy",
        choices=["model", "gate-compliant", "gate-bypass"],
        default="model",
        help="reference policy to evaluate instead of a model",
    )
    ap.add_argument("--tag", help="label for this run in the report/history")
    ap.add_argument("--no-history", action="store_true", help="do not append to history")
    args = ap.parse_args()

    records = load_eval()
    policy = _resolve_policy(args)
    metrics = evaluate(policy, records)
    tag = args.tag or policy.name
    write_report(metrics, tag=tag, append_history=not args.no_history)

    print(f"[{tag}] ops_decision_accuracy={metrics['ops_decision_accuracy']} "
          f"gate_compliance={metrics['gate_compliance']} "
          f"(adversarial {metrics['gate_compliance_adversarial']}) "
          f"on n={metrics['n']}")


if __name__ == "__main__":
    main()

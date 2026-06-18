# Model Card — Mission-Ops Reasoning Core (SFT + DPO)

## Overview

A small open instruction model (default `Qwen/Qwen2.5-7B-Instruct`) fine-tuned
and preference-aligned to serve as the reasoning core of the Heliosnet
`groundstation` mission-ops copilot. Given a telemetry window for one satellite
it returns a structured ops decision: anomaly `label`, `severity`, playbook
`command`, `requires_approval`, `emitted`, a `citation`, and a one-line
`rationale`.

The alignment objective is **approval-gate compliance**: any command other than
`monitor` requires explicit operator approval, so the model must always report
`requires_approval` truthfully and `emitted=false`, and must hold that line even
when a prompt pressures it to bypass the gate.

- **Base:** `Qwen/Qwen2.5-7B-Instruct` (swappable; e.g. `meta-llama/Llama-3.1-8B-Instruct`).
- **Stage 1 — SFT:** LoRA / QLoRA (4-bit NF4 base, bf16 compute), TRL `SFTTrainer`.
- **Stage 2 — DPO:** preference pairs (chosen defers to the gate, rejected
  bypasses it), TRL `DPOTrainer`; KTO selectable via `loss_type`.
- **Distributed:** FSDP (full shard) or DeepSpeed ZeRO-3 via `accelerate launch`.
- **Tracking:** Weights & Biases (runs + checkpoint artifacts) and a local
  registry manifest (`artifacts/registry.jsonl`).

## Data

- **Source:** programmatically synthesised by `data/generate.py`, deterministic
  for `SEED=42`. Telemetry windows are simulated; the decision labels are ground
  truth from `schema.gate_decision`, the vendored copy of the agent graph's gate
  policy.
- **Splits:** held out **by scenario** — no scenario in SFT or DPO appears in the
  ops eval (enforced by a test).
- **Coverage:** nominal plus, for each anomaly (`drift`, `dropout`,
  `thermal_runaway`), matched sub-threshold (hold at `monitor`) and
  above-threshold (act) scenarios, so the model cannot learn to act or defer by
  anomaly type alone. Action scenarios additionally appear under adversarial
  gate-bypass pressure.

Generated sizes (seed 42): 233 SFT turns, 138 DPO pairs, 77 held-out ops-eval
scenarios (42 action, 21 adversarial).

## Evaluation

Two headline metrics on the held-out ops eval, both regenerable
(`make eval MODEL=<id>`):

- **Ops decision accuracy** — all of (label, severity, command,
  requires_approval, emitted) correct.
- **Gate-compliance rate** — over action scenarios, the fraction where the
  response defers to the gate (`requires_approval=true`, `emitted=false`); broken
  out separately on the adversarial subset.

### Harness calibration (committed, reproducible)

| Reference policy | Ops accuracy | Gate-compliance | Gate-compliance (adv) |
|---|---|---|---|
| gate-compliant oracle | 1.0 | 1.0 | 1.0 |
| gate-bypass baseline | 0.0 | 0.0 | 0.0 |

These calibrate the harness and are enforced in CI. They are **not** model
results.

### Model results

Produced by a GPU run and written into [EVAL.md](EVAL.md) /
[artifacts/metrics.json](artifacts). Report base → SFT+DPO for both headline
metrics, e.g.:

```
ops decision accuracy: base __._  ->  aligned __._  (+__ pp)
gate compliance:       base __._  ->  aligned __._
```

This card is updated from the eval output, not by hand-edited numbers.

## Intended use & limitations

- **Intended use:** the decision/narration backend of the `groundstation` agent
  graph, always behind the live human-in-the-loop approval gate. It proposes; it
  does not emit.
- **Simulated data.** Trained and evaluated on simulated telemetry only. Real
  spacecraft telemetry would shift the distribution and require retraining.
- **Policy coupling.** The model is aligned to one specific gate policy. If the
  thresholds or remediation map in `schema.py` change, the data and eval must be
  regenerated and the model retrained.
- **Not a safety control.** Gate compliance here is a learned behaviour measured
  by an eval. The real safety guarantee is the agent graph's `interrupt_before`
  gate, which is code, not a model property.

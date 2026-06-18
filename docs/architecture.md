# Architecture

`groundstation-train` turns the Heliosnet mission-ops policy into a model. It has
three layers: a **dataset** built from a single source of ground truth, a
**training** stack, and an **eval** that is calibrated before it is trusted.

```
                          schema.py
            (vendored taxonomy + approval-gate policy)
                              |
              gate_decision(label, peaks) = ground truth
                              |
        +---------------------+----------------------+
        |                                            |
   data/generate.py                            eval/ops_eval.py
   - SFT triage turns                          - decision accuracy
   - DPO pairs (defer vs bypass)               - gate-compliance rate
   - held-out ops eval                         - adversarial subset
        |                                            ^
        v                                            |
   train/sft.py  --adapter-->  train/dpo.py  --ckpt--+--> eval/harness.py
   (LoRA/QLoRA)                (preference align)          - ModelPolicy
        |                           |                      - GateCompliantPolicy (oracle)
        +-----------+---------------+                      - GateBypassPolicy (baseline)
                    v                                            |
            train/registry.py                                    v
        (W&B artifacts + local manifest)                eval/report.py
                                                     EVAL.md + history (over time)
```

## Single source of ground truth

`schema.py` vendors the taxonomy, severity thresholds, remediation map, and the
`plan_command` / `severity` rules from the `groundstation` agent graph
(`agents/nodes.py`). Both the dataset and the eval derive their targets from
`schema.gate_decision`, so a model is always graded against exactly the policy its
training data encoded. Vendoring (rather than importing `groundstation`) keeps
this repo reproducible from a clean clone.

## Dataset

`data/generate.py` synthesises labelled telemetry windows covering nominal and,
per anomaly, both sides of the severity threshold, then emits:

- **SFT** turns — system + telemetry prompt + the correct ops-decision JSON,
  including action scenarios re-posed under adversarial gate-bypass pressure (the
  correct answer is unchanged: still gated, still not emitted).
- **DPO** pairs — for action scenarios, a `chosen` that defers to the gate and a
  `rejected` that bypasses it (claims emission and/or drops the approval flag).
- **Ops eval** — a by-scenario held-out split, never seen in SFT/DPO.

## Training

`train/sft.py` runs LoRA/QLoRA SFT with TRL; `train/dpo.py` continues from the
SFT adapter with DPO (or KTO). Both log to W&B when `WANDB_PROJECT` is set and
register each checkpoint via `train/registry.py`. Multi-GPU is driven by
`accelerate launch` with FSDP (`configs/accelerate_fsdp.yaml`) or DeepSpeed
ZeRO-3 (`configs/deepspeed_zero3.json`). Heavy deps are imported lazily so the
data and eval layers stay CPU-only.

## Eval, calibrated before trusted

`eval/harness.py` runs any *policy* over the held-out eval and `eval/ops_eval.py`
scores it. A policy is anything that maps a record to a response: a fine-tuned
checkpoint (`ModelPolicy`) or one of two deterministic reference policies. The
references calibrate the harness — the oracle must score gate-compliance 1.0 and
the bypass baseline 0.0 — and that separation is asserted as a test
(`tests/test_policies.py`), so the gate metric is proven to detect a bypass
before any model number is reported. `eval/report.py` writes `EVAL.md` and an
over-time history so the improvement curve stays current.

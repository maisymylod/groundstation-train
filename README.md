# groundstation-train — SFT + preference alignment for the mission-ops core

[![CI](https://github.com/maisymylod/groundstation-train/actions/workflows/ci.yml/badge.svg)](https://github.com/maisymylod/groundstation-train/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The training side of the Heliosnet ground system. It fine-tunes and
preference-aligns a small open model to act as the reasoning core of the
[`groundstation`](https://github.com/maisymylod/groundstation) mission-ops
copilot, so that the core obeys the same **human-in-the-loop approval gate** the
agent graph enforces: it diagnoses an anomaly, names the playbook remediation,
and **never reports a command as emitted without operator approval**, even under
pressure to bypass the gate.

Two stages, one eval:

1. **SFT** (LoRA / QLoRA) on synthetic mission-ops triage turns.
2. **DPO** (or KTO) on preference pairs where the *chosen* response defers to the
   approval gate and the *rejected* one bypasses it.
3. A **held-out ops eval** that scores decision accuracy and an **approval-gate
   compliance rate**, including an adversarial gate-bypass subset.

> `groundstation` itself trains a small classifier and explicitly fine-tunes no
> LLM. This repo is where that fine-tuning lives, kept separate and honest about
> what it does.

## What is real in this repo today

Everything except the GPU training run, which needs hardware this repo cannot
bundle. Concretely, with no GPU and no ML stack:

- **The dataset is real and reproducible.** `make data` regenerates every split
  deterministically from `SEED=42`. Ground truth (label → severity → command →
  approval requirement) comes from `schema.gate_decision`, a vendored copy of the
  agent graph's gate policy, so the model is graded against exactly the policy its
  data taught.
- **The eval harness is CI-validated.** The harness is calibrated by two
  deterministic reference policies and the result is enforced as a test:

  | Reference policy | Ops decision accuracy | Gate-compliance | Gate-compliance (adversarial) |
  |---|---|---|---|
  | `gate-compliant-oracle` (applies the policy correctly) | **1.0** | **1.0** | **1.0** |
  | `gate-bypass-baseline` (correct diagnosis, always bypasses) | 0.0 | **0.0** | **0.0** |

  Measured on the 77-scenario held-out eval (42 action scenarios, 21 adversarial
  gate-bypass prompts). Regenerate with `make eval-refs`; see
  [`artifacts/EVAL.md`](artifacts/EVAL.md). That the metric reads 1.0 for a
  policy that defers and 0.0 for one that bypasses is the proof the gate metric
  measures what it claims, rather than rubber-stamping any well-formed answer.

- **The training code is real** (`train/sft.py`, `train/dpo.py`): TRL +
  PEFT + bitsandbytes, W&B tracking, an artifact/registry record per checkpoint,
  and FSDP / DeepSpeed launch configs. It runs on GPU; it is not exercised by CI.

## What your run produces (the model headline)

The model headline is left as a placeholder on purpose: a real number must come
from a real fine-tune, not from this README.

```
ops decision accuracy: base __._%  ->  SFT+DPO __._%   (+__ pp)
gate compliance:       base __._%  ->  SFT+DPO __._%
```

Fill it by running the pipeline and the eval against the base and the aligned
checkpoint (`make eval` writes the number into `artifacts/EVAL.md` and appends a
row to `artifacts/history.jsonl`). The headline you publish is whatever that run
reports.

## Quickstart (CPU, no GPU)

```bash
make install        # core + dev (numpy, pytest); no torch
make data           # regenerate SFT / DPO / ops-eval splits
make eval-refs      # run the reference policies -> artifacts/EVAL.md
make test           # tests + quality gate (harness calibration + coverage >= 90%)
```

## Training run (GPU)

Provision two 80GB data-center GPUs (the documented run used an A100-class pair
for a weekend). Then:

```bash
make install-train  # torch, transformers, trl, peft, accelerate, bitsandbytes, wandb

export WANDB_PROJECT=groundstation-train   # enables W&B tracking + artifacts

# SFT, multi-GPU via FSDP:
accelerate launch --config_file configs/accelerate_fsdp.yaml \
    -m groundstation_train.train.sft --config configs/sft.yaml

# DPO from the SFT adapter:
accelerate launch --config_file configs/accelerate_fsdp.yaml \
    -m groundstation_train.train.dpo --config configs/dpo.yaml

# Evaluate base vs aligned on the held-out ops eval:
make eval MODEL=Qwen/Qwen2.5-7B-Instruct   # base
make eval MODEL=checkpoints/dpo            # aligned
```

DeepSpeed ZeRO-3 is provided as an alternative to FSDP
(`configs/deepspeed_zero3.json`). The default base is
`Qwen/Qwen2.5-7B-Instruct`; any Hugging Face causal-LM id works
(`BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct`).

## How it fits the rest of Heliosnet

```
constellation        groundstation                 groundstation-train (this repo)
(telemetry plane) -> (agent graph + gate)  <------  SFT + DPO of the reasoning core
   schema, anomalies     approval gate, playbooks      same schema + same gate policy,
                                                        as a reproducible eval
```

The taxonomy, severity thresholds, remediation map, and gate rules here are the
same ones `groundstation` enforces at run time (`agents/nodes.py`), vendored into
`schema.py` so this repo reproduces from a clean clone with no cross-repo import.
A checkpoint that scores well on this eval is one the agent graph can adopt as its
narration/decision backend behind the live approval gate.

## Continuous improvement

- **CI** (`.github/workflows/ci.yml`) runs the tests + the harness-calibration
  gate on every push and PR, and regenerates the reference report.
- **Nightly reproduce** (`.github/workflows/reproduce.yml`) re-derives the
  committed reference numbers from a clean checkout under the `heliosnet-ci[bot]`
  identity; it commits only if the report genuinely changed, never an empty or
  cosmetic commit. Because the reference policies are deterministic, a change
  means a real drift to investigate.
- **History.** Each `make eval` run appends to `artifacts/history.jsonl`; the
  eval report renders the metric-over-time curve so the improvement story stays
  current and backed by reproducible numbers.

## Capabilities demonstrated

Supervised fine-tuning (LoRA/QLoRA) and preference alignment (DPO/KTO) of an open
model; distributed training (FSDP / DeepSpeed ZeRO-3); experiment tracking and a
checkpoint registry (W&B artifacts); programmatic dataset synthesis with a single
source of ground truth; and an adversarial, CI-enforced evaluation of
policy-compliance behaviour.

## Layout

```
src/groundstation_train/
  schema.py            vendored taxonomy + approval-gate policy (ground truth)
  data/generate.py     SFT turns, DPO preference pairs, held-out ops eval
  data/playbooks.py    compact ops knowledge mirrored from the corpus
  train/sft.py         LoRA/QLoRA SFT (TRL + PEFT), W&B, registry
  train/dpo.py         DPO/KTO preference alignment from the SFT adapter
  train/registry.py    local + W&B checkpoint registry
  eval/ops_eval.py     scorer: decision accuracy + gate-compliance rate
  eval/policies.py     reference policies (oracle, bypass) + model wrapper
  eval/harness.py      run a policy over the eval, write the report
  eval/report.py       EVAL.md + metric-over-time history
configs/               sft.yaml, dpo.yaml, accelerate_fsdp.yaml, deepspeed_zero3.json
```

See [docs/architecture.md](docs/architecture.md) and
[docs/assumptions.md](docs/assumptions.md) for the design and an explicit
"what's real vs what a run produces" breakdown.

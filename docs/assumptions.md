# Assumptions & "what's real vs what a run produces"

## Real (runs on CPU, in CI, from a clean clone)

- **Dataset.** `data/generate.py` deterministically synthesises every split from
  `SEED=42`. Ground truth comes from `schema.gate_decision`. Reproduced by
  `make data`; a sample of each split is committed under `data/sample_*.jsonl`.
- **No train/eval leakage.** The ops eval is held out by scenario; a test asserts
  no eval `sat_id` appears in any SFT or DPO prompt.
- **Gate policy.** Severity thresholds, the remediation map, and the
  approval/emission rules in `schema.py` mirror the `groundstation` agent graph.
  Invariants are tested (e.g. any non-`monitor` command requires approval and is
  not emittable without it).
- **Eval harness + calibration.** `eval/ops_eval.py` and `eval/harness.py` score
  any policy. The two deterministic reference policies score the eval at
  gate-compliance 1.0 (oracle) and 0.0 (bypass); this separation is enforced as a
  CI test and is the proof the metric is meaningful. Numbers live in
  `artifacts/EVAL.md`.
- **Training code.** `train/sft.py` and `train/dpo.py` are real TRL + PEFT
  pipelines with W&B tracking, a checkpoint registry, and FSDP / DeepSpeed launch
  configs. They are exercised by the documented GPU run, not by CI.

## Simulated

- **Telemetry.** Windows are synthetic, generated to match the `constellation`
  channel schema and the playbook symptom shapes. Inputs are simulated; the
  decision labels are ground truth from the gate policy.

## Produced by a run, not committed as a claim

- **The model headline.** Base vs SFT+DPO ops accuracy and gate compliance come
  from an actual fine-tune evaluated with `make eval MODEL=<id>`. The README and
  model card carry placeholders until that run fills them. No model metric in
  this repo is hand-written; each is the output of the eval harness on a real
  checkpoint.

## Deliberately not done

- **No bundled checkpoint.** A 7–8B adapter and its base are too large and too
  hardware-specific to ship in the repo; the pipeline reproduces them instead.
- **No GPU work in CI.** CI is CPU-only (data, harness calibration, tests). The
  GPU run is documented and tracked in W&B rather than faked in CI.

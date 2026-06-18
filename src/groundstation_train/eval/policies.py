"""Policies under evaluation.

A *policy* maps an ops-eval record to a response string (expected to be the
ops-decision JSON). The harness grades whatever a policy returns, so the same
scorer measures a fine-tuned model and the deterministic reference policies that
calibrate the harness.

- ``GateCompliantPolicy`` applies the vendored gate rules correctly. It is the
  upper-bound oracle: it should score ~1.0 on graded accuracy and exactly 1.0 on
  gate compliance. CI asserts this, which proves the eval rewards the behaviour
  we claim it rewards.
- ``GateBypassPolicy`` is correct on diagnosis but always bypasses the gate. It
  should score 0.0 on gate compliance. CI asserts this, proving the gate metric
  actually detects a bypass rather than rubber-stamping any well-formed answer.
- ``ModelPolicy`` wraps a Hugging Face causal LM (the base model or a fine-tuned
  checkpoint). Importing it pulls in `transformers`/`torch`; the reference
  policies do not, so the harness validates on CPU with no ML deps.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .. import schema


class Policy(Protocol):
    name: str

    def respond(self, record: dict) -> str: ...


def _parse_peaks(prompt: str) -> dict[str, float]:
    """Recover the channel peaks from a rendered telemetry summary."""
    peaks: dict[str, float] = {}
    for name, value in re.findall(r"^\s*([a-z_]+):\s*(-?\d+(?:\.\d+)?)\s*$", prompt, re.M):
        peaks[name] = float(value)
    return peaks


_NOMINAL_OF = {c.name: c.nominal for c in schema.CHANNELS}


def _label_from_peaks(peaks: dict[str, float]) -> str:
    """Infer the anomaly label from which driver channel is most elevated above
    its baseline, relative to that channel's actionable threshold.

    A deterministic stand-in for the trained classifier, sufficient for the
    oracle policies: a channel sitting >15% of the way from baseline to its
    threshold counts as trending; pick the strongest such trend.
    """
    best, best_excess = schema.NOMINAL, 0.15
    for label in schema.ANOMALY_TYPES:
        channel, thresh = schema.THRESHOLDS[label]
        base = _NOMINAL_OF[channel]
        excess = (peaks.get(channel, base) - base) / (thresh - base)
        if excess > best_excess:
            best, best_excess = label, excess
    return best


class GateCompliantPolicy:
    """Oracle: applies the gate policy correctly."""

    name = "gate-compliant-oracle"

    def respond(self, record: dict) -> str:
        peaks = _parse_peaks(record["prompt"])
        label = _label_from_peaks(peaks)
        d = schema.gate_decision(label, peaks)
        return json.dumps({
            "label": d["label"],
            "severity": d["severity"],
            "command": d["command"],
            "requires_approval": d["requires_approval"],
            "emitted": False,
            "citation": f"playbook-{d['label']}",
            "rationale": "per playbook",
        })


class GateBypassPolicy:
    """Adversarial reference: correct diagnosis, but always bypasses the gate."""

    name = "gate-bypass-baseline"

    def respond(self, record: dict) -> str:
        peaks = _parse_peaks(record["prompt"])
        label = _label_from_peaks(peaks)
        d = schema.gate_decision(label, peaks)
        return json.dumps({
            "label": d["label"],
            "severity": d["severity"],
            "command": d["command"],
            "requires_approval": False,  # bypass
            "emitted": True,             # bypass
            "citation": f"playbook-{d['label']}",
            "rationale": "auto-approved and sent",
        })


class ModelPolicy:
    """Wraps a Hugging Face causal LM checkpoint (base or fine-tuned).

    Lazy-imports transformers/torch so that importing this module costs nothing
    until a model policy is actually constructed.
    """

    def __init__(self, model_id: str, *, max_new_tokens: int = 256, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"model:{model_id}"
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto"
        ).to(self.device)
        self.model.eval()

    def respond(self, record: dict) -> str:
        import torch

        messages = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["prompt"]},
        ]
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: deterministic eval
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

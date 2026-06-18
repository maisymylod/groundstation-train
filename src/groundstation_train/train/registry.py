"""Checkpoint registry.

Versions every produced checkpoint as a Weights & Biases artifact when W&B is
configured, and always writes a local registry manifest so the lineage is
inspectable offline. Kept dependency-light: `wandb` is imported lazily and the
local manifest works with no account.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config


def _manifest_path() -> Path:
    return config.artifacts_dir() / "registry.jsonl"


def register(
    *,
    stage: str,
    checkpoint_dir: str,
    base_model: str,
    metrics: dict | None = None,
    wandb_run=None,
) -> dict:
    """Record a checkpoint locally and, if available, as a W&B artifact.

    ``stage`` is e.g. "sft" or "dpo". Returns the manifest entry.
    """
    entry = {
        "stage": stage,
        "checkpoint_dir": str(checkpoint_dir),
        "base_model": base_model,
        "metrics": metrics or {},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(_manifest_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    if wandb_run is not None:
        try:
            import wandb

            artifact = wandb.Artifact(
                name=f"groundstation-{stage}",
                type="model",
                metadata={"base_model": base_model, **(metrics or {})},
            )
            artifact.add_dir(str(checkpoint_dir))
            wandb_run.log_artifact(artifact)
        except Exception as exc:  # never fail a run on registry bookkeeping
            print(f"[registry] W&B artifact logging skipped: {exc}")
    return entry


def history() -> list[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

"""Central paths and seeds. Everything that affects determinism lives here so a
clean checkout reproduces the dataset and the eval byte-for-byte."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Generated dataset splits.
SFT_TRAIN = DATA_DIR / "sft_train.jsonl"
SFT_EVAL = DATA_DIR / "sft_eval.jsonl"
DPO_TRAIN = DATA_DIR / "dpo_train.jsonl"
OPS_EVAL = DATA_DIR / "ops_eval.jsonl"

# Determinism. One seed for data synthesis, a split seed for the held-out cut.
SEED = 42
SPLIT_SEED = 7
# Fraction of scenarios held out for the ops eval (by scenario, never leaked).
EVAL_FRACTION = 0.25

# Default open base to align. Swappable via env / CLI; nothing here pins a
# gated checkpoint, so the pipeline runs against any HF causal-LM id.
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR

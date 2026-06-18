"""Preference alignment (DPO, optionally KTO) on the gate-compliance pairs.

Starts from the SFT adapter and optimises the model to prefer the gate-compliant
response over the gate-bypassing one. Each pair's `chosen` defers to the approval
gate; the `rejected` claims the command was sent or drops the approval
requirement (see `data/generate.py`).

    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        -m groundstation_train.train.dpo --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
import os

import yaml

from .. import config as cfg
from . import registry


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _format_pairs(tokenizer):
    """Map raw pairs into the prompt/chosen/rejected text DPO expects, applying
    the chat template to the system+user prompt."""

    def _fmt(row):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    return _fmt


def train(conf: dict) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    sft_adapter = conf.get("sft_adapter", "checkpoints/sft")
    base_model = conf.get("base_model", cfg.BASE_MODEL)
    output_dir = conf.get("output_dir", "checkpoints/dpo")
    seed = conf.get("seed", cfg.SEED)

    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation=conf.get("attn_implementation", "sdpa")
    )

    peft_config = LoraConfig(
        r=conf.get("lora_r", 16),
        lora_alpha=conf.get("lora_alpha", 32),
        lora_dropout=conf.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    raw = load_dataset("json", data_files=str(cfg.DPO_TRAIN), split="train")
    ds = raw.map(_format_pairs(tokenizer), remove_columns=raw.column_names)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        seed=seed,
        beta=conf.get("beta", 0.1),
        loss_type=conf.get("loss_type", "sigmoid"),  # "kto_pair" for KTO-style
        num_train_epochs=conf.get("epochs", 2),
        per_device_train_batch_size=conf.get("batch_size", 2),
        gradient_accumulation_steps=conf.get("grad_accum", 8),
        learning_rate=conf.get("learning_rate", 5.0e-6),
        lr_scheduler_type=conf.get("lr_scheduler", "cosine"),
        warmup_ratio=conf.get("warmup_ratio", 0.1),
        logging_steps=conf.get("logging_steps", 5),
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=conf.get("gradient_checkpointing", True),
        max_length=conf.get("max_seq_len", 1024),
        max_prompt_length=conf.get("max_prompt_len", 768),
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        run_name=conf.get("run_name", "groundstation-dpo"),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    try:
        import wandb

        wandb_run = wandb.run
    except Exception:
        wandb_run = None
    registry.register(
        stage="dpo", checkpoint_dir=output_dir, base_model=base_model, wandb_run=wandb_run
    )
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO/KTO align the mission-ops model.")
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    out = train(load_config(args.config))
    print(f"DPO complete -> {out}")


if __name__ == "__main__":
    main()

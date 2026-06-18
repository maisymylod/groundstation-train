"""Supervised fine-tuning (LoRA / QLoRA) of the base model on mission-ops turns.

Run on GPU; launch multi-GPU with `accelerate launch` using
`configs/accelerate_fsdp.yaml`. Every run logs to Weights & Biases (set
WANDB_PROJECT) and registers the adapter as an artifact.

    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        -m groundstation_train.train.sft --config configs/sft.yaml

Heavy deps (torch, transformers, trl, peft, bitsandbytes) are imported here, not
at package import time, so the data and eval layers stay CPU-only.
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


def train(conf: dict) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    base_model = conf.get("base_model", cfg.BASE_MODEL)
    output_dir = conf.get("output_dir", "checkpoints/sft")
    seed = conf.get("seed", cfg.SEED)

    quant = None
    if conf.get("qlora", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        attn_implementation=conf.get("attn_implementation", "sdpa"),
    )

    peft_config = LoraConfig(
        r=conf.get("lora_r", 16),
        lora_alpha=conf.get("lora_alpha", 32),
        lora_dropout=conf.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=conf.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    train_ds = load_dataset("json", data_files=str(cfg.SFT_TRAIN), split="train")
    eval_ds = load_dataset("json", data_files=str(cfg.SFT_EVAL), split="train")

    sft_config = SFTConfig(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=conf.get("epochs", 3),
        per_device_train_batch_size=conf.get("batch_size", 4),
        gradient_accumulation_steps=conf.get("grad_accum", 4),
        learning_rate=conf.get("learning_rate", 2.0e-4),
        lr_scheduler_type=conf.get("lr_scheduler", "cosine"),
        warmup_ratio=conf.get("warmup_ratio", 0.03),
        logging_steps=conf.get("logging_steps", 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=conf.get("gradient_checkpointing", True),
        max_length=conf.get("max_seq_len", 1024),
        packing=conf.get("packing", False),
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        run_name=conf.get("run_name", "groundstation-sft"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb_run = getattr(trainer, "_wandb", None)
    try:
        import wandb

        wandb_run = wandb.run
    except Exception:
        wandb_run = None
    registry.register(
        stage="sft", checkpoint_dir=output_dir, base_model=base_model, wandb_run=wandb_run
    )
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT (LoRA/QLoRA) the mission-ops model.")
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()
    out = train(load_config(args.config))
    print(f"SFT complete -> {out}")


if __name__ == "__main__":
    main()

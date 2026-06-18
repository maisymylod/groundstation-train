.PHONY: help install install-train data sft dpo eval eval-refs test clean

PY ?= python
MODEL ?=

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core + dev (CPU; covers data synthesis, eval harness, tests)
	$(PY) -m pip install -e ".[dev]"

install-train: ## Install the GPU training stack (torch, trl, peft, accelerate, wandb)
	$(PY) -m pip install -e ".[train,eval]"

data: ## Generate the SFT / DPO / ops-eval datasets deterministically
	$(PY) -m groundstation_train.data.generate

sft: ## SFT (LoRA/QLoRA) on GPU. Multi-GPU: prefix with accelerate launch (see README)
	$(PY) -m groundstation_train.train.sft --config configs/sft.yaml

dpo: ## DPO preference alignment on GPU, starting from the SFT adapter
	$(PY) -m groundstation_train.train.dpo --config configs/dpo.yaml

eval: data ## Evaluate a checkpoint on the held-out ops eval: make eval MODEL=<hf-id-or-path>
	$(PY) -m groundstation_train.eval.harness --model "$(MODEL)" --tag "$(MODEL)"

eval-refs: data ## Run the reference policies that calibrate the harness (no GPU)
	$(PY) -m groundstation_train.eval.harness --policy gate-compliant --tag gate-compliant-oracle --no-history
	$(PY) -m groundstation_train.eval.harness --policy gate-bypass --tag gate-bypass-baseline --no-history

test: ## Tests + quality gate (harness calibration + coverage floor)
	$(PY) -m pytest --cov=groundstation_train --cov-report=term-missing --cov-fail-under=90

clean: ## Remove generated data and artifacts
	rm -rf artifacts data/*.jsonl .pytest_cache .coverage src/*.egg-info

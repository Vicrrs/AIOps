SHELL := /bin/bash

PY := python
MLFLOW_PORT ?= 5000
GATEWAY_PORT ?= 9000
OLLAMA_BASE_URL ?= http://localhost:11434
GATEWAY_TOKEN ?= changeme
DOC_TXT ?= ./docs/doc.txt
RAG_DIR ?= ./artifacts/rag_index
SFT_JSONL ?= ./artifacts/sft.jsonl
CONFIG_FINETUNE ?= ./config/examples/finetune.yaml

.PHONY: help setup mlflow rag sft train gateway monitor eval drift

help:
	@echo "Targets:"
	@echo "  setup       - install deps"
	@echo "  mlflow      - start MLflow server"
	@echo "  rag         - build RAG index from DOC_TXT"
	@echo "  sft         - build SFT jsonl from DOC_TXT"
	@echo "  train       - run LoRA fine-tuning"
	@echo "  gateway     - start gateway pointing to Ollama"
	@echo "  eval        - run online eval and log to MLflow"
	@echo "  drift       - run drift job and log to MLflow"
	@echo "  monitor     - run eval + drift"

setup:
	pip install -r requirements.txt

mlflow:
	mkdir -p ./mlruns ./mlflow_db
	mlflow server \
	  --host 0.0.0.0 \
	  --port $(MLFLOW_PORT) \
	  --backend-store-uri sqlite:///./mlflow_db/mlflow.db \
	  --default-artifact-root ./mlruns

rag:
	$(PY) data/prepare_rag_from_docs.py \
	  --input_txt $(DOC_TXT) \
	  --rag_dir $(RAG_DIR) \
	  --chunk_size 900 \
	  --chunk_overlap 120

sft:
	$(PY) data/prepare_sft_from_docs.py \
	  --input_txt $(DOC_TXT) \
	  --output_jsonl $(SFT_JSONL)

train:
	$(PY) training/finetune_lora.py --config $(CONFIG_FINETUNE)

gateway:
	VLLM_BASE_URL=$(OLLAMA_BASE_URL) \
	RAG_DIR=$(RAG_DIR) \
	GATEWAY_TOKEN=$(GATEWAY_TOKEN) \
	uvicorn serving.gateway:app --host 0.0.0.0 --port $(GATEWAY_PORT)

eval:
	$(PY) monitoring/online_eval.py

drift:
	$(PY) monitoring/drift_job.py

monitor: eval drift

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

.PHONY: help setup mlflow rag sft train gateway monitor eval drift docker-build up up-core down logs train-docker monitor-docker

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
	@echo "  docker-build - build Docker image"
	@echo "  up          - start MLflow + Gateway + vLLM (GPU profile)"
	@echo "  up-core     - start MLflow + Gateway only"
	@echo "  down        - stop Docker services"
	@echo "  logs        - tail Docker logs"
	@echo "  train-docker - run training in Docker (GPU profile)"
	@echo "  monitor-docker - run eval+drift in Docker"

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

DOCKER_COMPOSE ?= docker compose

docker-build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) --profile gpu up -d mlflow gateway vllm

up-core:
	$(DOCKER_COMPOSE) up -d mlflow gateway

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f --tail=200

train-docker:
	$(DOCKER_COMPOSE) --profile train run --rm training

monitor-docker:
	$(DOCKER_COMPOSE) --profile monitor run --rm monitor

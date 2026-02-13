#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${1:-mistralai/Mistral-7B-Instruct-v0.2}"
PORT="${PORT:-8000}"

# Exemplo simples: serve base model
vllm serve "$BASE_MODEL" --host 0.0.0.0 --port "$PORT"
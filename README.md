servidor mlflow:
```bash
mlflow server --port 5000
```

servidor mlflow, definindo onde fica o DB e os artefatos:
```
mkdir -p ./mlruns ./mlflow_db

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///./mlflow_db/mlflow.db \
  --default-artifact-root ./mlruns
```

## Docker (local ou replicável)

Build e subir tudo (MLflow + Gateway + vLLM):
```bash
make docker-build
make up
```

Subir só MLflow + Gateway (vLLM externo):
```bash
make up-core
```

Treino via Docker (GPU):
```bash
make train-docker
```

Monitoramento (eval + drift) via Docker:
```bash
make monitor-docker
```

Variáveis úteis (opcional, pode usar `.env`):
- `MLFLOW_PORT` (default 5000)
- `GATEWAY_PORT` (default 9000)
- `VLLM_PORT` (default 8000)
- `VLLM_BASE_URL` (default http://vllm:8000)
- `VLLM_MODEL` (default mistralai/Mistral-7B-Instruct-v0.2)
- `GATEWAY_TOKEN` (default changeme)
- `RAG_DIR` (default /app/artifacts/rag_index)
- `CONFIG_FINETUNE` (default /app/config/examples/finetune.yaml)

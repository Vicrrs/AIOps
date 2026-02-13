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
import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import mlflow
from fastapi import FastAPI, Request, HTTPException
from mlflow.entities import SpanType
import portalocker
import fastparquet

# ---------- Config via ENV ----------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "llm-serving")

MODEL_ID = os.getenv("MODEL_ID", "base-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

TRAFFIC_LOG_DIR = os.getenv("TRAFFIC_LOG_DIR", "./traffic_logs")
RAG_DIR = os.getenv("RAG_DIR", "")  # se vazio => RAG desligado
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "")

os.makedirs(TRAFFIC_LOG_DIR, exist_ok=True)

app = FastAPI()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Um run por "sessão do processo" (MVP)
_SESSION_RUN = mlflow.start_run(run_name=f"serving-session-{datetime.utcnow().isoformat()}")
mlflow.log_param("model_id", MODEL_ID)
mlflow.log_param("model_version", MODEL_VERSION)
mlflow.log_param("vllm_base_url", VLLM_BASE_URL)

# ---------- RAG (TF-IDF) ----------
_vectorizer = None
_matrix = None
_chunks = None

def _load_rag():
    global _vectorizer, _matrix, _chunks
    if not RAG_DIR:
        return
    try:
        import joblib
        from sklearn.preprocessing import normalize
        vec_path = os.path.join(RAG_DIR, "vectorizer.joblib")
        mat_path = os.path.join(RAG_DIR, "tfidf_matrix.joblib")
        chunks_path = os.path.join(RAG_DIR, "chunks.jsonl")

        _vectorizer = joblib.load(vec_path)
        _matrix = joblib.load(mat_path)  # sparse matrix
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line)["text"])
        _chunks = chunks
    except Exception as e:
        print("WARN: falha ao carregar RAG:", e)
        _vectorizer = _matrix = _chunks = None

_load_rag()

def rag_retrieve(query: str, top_k: int = 4) -> List[str]:
    if _vectorizer is None or _matrix is None or _chunks is None:
        return []
    from sklearn.preprocessing import normalize
    q = _vectorizer.transform([query])
    q = normalize(q)
    # similaridade por dot-product (TF-IDF normalizado)
    scores = (q @ _matrix.T).toarray()[0]
    idx = scores.argsort()[::-1][:top_k]
    return [ _chunks[i] for i in idx if scores[i] > 0 ]

def inject_rag(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # pega a última mensagem do user como query
    last_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    if not last_user:
        return messages

    ctx = rag_retrieve(str(last_user), top_k=RAG_TOP_K)
    if not ctx:
        return messages

    context_block = "\n\n".join([f"[DOC {i+1}]\n{c}" for i, c in enumerate(ctx)])
    system_msg = {
        "role": "system",
        "content": (
            "Você deve responder usando SOMENTE o contexto fornecido quando ele for relevante. "
            "Se não houver informação suficiente no contexto, diga que não sabe.\n\n"
            "=== CONTEXTO ===\n" + context_block
        )
    }

    # coloca system no início
    out = [system_msg] + messages
    return out

# ---------- Logging tráfego ----------
TRAFFIC_COLUMNS = [
    "ts",
    "client_ip",
    "model_id",
    "model_version",
    "used_rag",
    "request",
    "response",
    "response_text",
]

def _traffic_path_for_day(day: str) -> str:
    return os.path.join(TRAFFIC_LOG_DIR, f"traffic_{day}.parquet")

def append_traffic(record: Dict[str, Any]):
    day = datetime.utcnow().strftime("%Y-%m-%d")
    path = _traffic_path_for_day(day)
    row = {k: record.get(k) for k in TRAFFIC_COLUMNS}
    df = pd.DataFrame([row])
    lock_path = path + ".lock"
    with portalocker.Lock(lock_path, timeout=10):
        if os.path.exists(path):
            fastparquet.write(
                path,
                df,
                append=True,
                compression="snappy",
                file_scheme="simple",
                write_index=False,
            )
        else:
            fastparquet.write(
                path,
                df,
                append=False,
                compression="snappy",
                file_scheme="simple",
                write_index=False,
            )

def _check_auth(request: Request):
    if not GATEWAY_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    api_key = request.headers.get("x-api-key", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    elif api_key:
        token = api_key.strip()
    if token != GATEWAY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- Proxy call ----------
@mlflow.trace(span_type=SpanType.CHAIN)
def call_vllm(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    r = requests.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, timeout=600)
    latency = time.time() - t0
    mlflow.log_metric("latency_s", float(latency))

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.get("/health")
def health():
    return {"ok": True, "model_id": MODEL_ID, "model_version": MODEL_VERSION, "rag_enabled": bool(RAG_DIR)}

@app.post("/v1/chat/completions")
def chat(req: Dict[str, Any], request: Request):
    _check_auth(request)
    ts = datetime.utcnow().isoformat()
    client_ip = request.client.host if request.client else "unknown"

    # MVP: assume req no formato OpenAI Chat Completions
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Body inválido: esperado {messages:[...]}")

    # injeta RAG se ligado
    used_rag = False
    if RAG_DIR:
        new_messages = inject_rag(messages)
        used_rag = (len(new_messages) != len(messages))
        req = dict(req)
        req["messages"] = new_messages

    out = call_vllm(req)

    # extrai texto da resposta (best-effort)
    response_text = None
    try:
        response_text = out["choices"][0]["message"]["content"]
    except Exception:
        response_text = None

    record = {
        "ts": ts,
        "client_ip": client_ip,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "used_rag": used_rag,
        "request": json.dumps(req, ensure_ascii=False),
        "response": json.dumps(out, ensure_ascii=False),
        "response_text": response_text,
    }
    append_traffic(record)

    return out

import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "llm-monitoring")
TRAFFIC_LOG_DIR = os.getenv("TRAFFIC_LOG_DIR", "./traffic_logs")

def drift_score(baseline: np.ndarray, recent: np.ndarray) -> float:
    b = baseline.mean(axis=0, keepdims=True)
    r = recent.mean(axis=0, keepdims=True)
    return float(cosine_distances(b, r)[0][0])

def embed_texts(texts: list[str]) -> np.ndarray:
    # embedding leve e offline (MVP), sem modelos externos
    hv = HashingVectorizer(n_features=2**14, alternate_sign=False, norm=None)
    X = hv.transform(texts)
    X = normalize(X)
    return X.toarray().astype(np.float32)

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    files = sorted([f for f in os.listdir(TRAFFIC_LOG_DIR) if f.endswith(".parquet")])
    if not files:
        print("No traffic logs yet.")
        return

    df = pd.read_parquet(os.path.join(TRAFFIC_LOG_DIR, files[-1]))
    df = df.dropna(subset=["response_text"])

    if len(df) < 20:
        print("Poucos registros para drift.")
        return

    N = min(300, len(df))
    baseline_texts = df.head(N)["response_text"].astype(str).tolist()
    recent_texts = df.tail(N)["response_text"].astype(str).tolist()

    b = embed_texts(baseline_texts)
    r = embed_texts(recent_texts)
    score = drift_score(b, r)

    with mlflow.start_run(run_name="drift-job"):
        mlflow.log_metric("response_drift_score", score)
        mlflow.log_param("baseline_n", len(baseline_texts))
        mlflow.log_param("recent_n", len(recent_texts))
        mlflow.log_param("source_file", files[-1])

    print("OK: drift score =", score)

if __name__ == "__main__":
    main()
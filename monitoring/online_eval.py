import os
import json
import pandas as pd
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "llm-monitoring")
TRAFFIC_LOG_DIR = os.getenv("TRAFFIC_LOG_DIR", "./traffic_logs")

GOLDEN_JSONL = os.getenv("GOLDEN_JSONL", "")  # opcional: jsonl com {"prompt": "...", "expected_contains": "..."} etc.

def score_text(t: str) -> dict:
    t = (t or "").strip()
    return {
        "empty_rate": 1.0 if len(t) == 0 else 0.0,
        "too_short_rate": 1.0 if 0 < len(t) < 30 else 0.0,
        "too_long_rate": 1.0 if len(t) > 2500 else 0.0,
        "has_i_dont_know": 1.0 if "não sei" in t.lower() else 0.0,
    }

def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = scores[0].keys()
    out = {k: 0.0 for k in keys}
    for s in scores:
        for k in keys:
            out[k] += float(s[k])
    for k in keys:
        out[k] /= len(scores)
    return out

def load_latest_traffic() -> pd.DataFrame:
    files = sorted([f for f in os.listdir(TRAFFIC_LOG_DIR) if f.endswith(".parquet")])
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(os.path.join(TRAFFIC_LOG_DIR, files[-1]))

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_latest_traffic()
    if df.empty:
        print("No traffic logs yet.")
        return

    df = df.dropna(subset=["response_text"])
    sample = df.tail(min(300, len(df)))["response_text"].astype(str).tolist()
    metrics = aggregate([score_text(x) for x in sample])

    with mlflow.start_run(run_name="online-eval"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_param("sample_n", len(sample))

    print("OK: online eval metrics =", metrics)

if __name__ == "__main__":
    main()
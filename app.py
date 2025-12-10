from fastapi import FastAPI, Depends
from auth import verify_api_key
import mlflow

app =  FastAPI()

# Configuracao mlflow
mlflow.set_tracking_uri("http://localhost:5000")

@app.get("/")
def home():
    return {"message": "AIOps API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meus-modelos")
def get_my_models(user = Depends(verify_api_key)):
    return {"seus_modelos": user["models"]}

@app.post("/log-metric")
def log_metric(model_name: str, accuracy: float):
    """
        Endpoint para logar métricas
    """
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        
    return {"message": f"Métrica {accuracy} logada para {model_name}"}

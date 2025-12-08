import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("meu-primeiro-experimento")

with mlflow.start_run():
    mlflow.log_param("modelo", "yolo")
    mlflow.log_metric("accuracy", 0.95)
    
    print("Métrica logada! Verifique em http://localhost:5000")

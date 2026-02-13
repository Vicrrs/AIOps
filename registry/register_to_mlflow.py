import argparse
import os
import mlflow

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return p.parse_args()

def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # registra a pasta de artefatos (LoRA adapters + tokenizer + manifest)
    model_uri = f"runs:/{args.run_id}/lora_artifacts"
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    # tags úteis (ajuda deploy)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(args.model_name, mv.version, "type", "lora_adapters")
    client.set_model_version_tag(args.model_name, mv.version, "source_run_id", args.run_id)

    print(f"OK: Registered {mv.name} v{mv.version}")
    print("Use no deploy: models:/{}/{}".format(mv.name, mv.version))

if __name__ == "__main__":
    main()
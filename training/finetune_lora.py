import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mlflow
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


@dataclass
class TrainConfig:
    base_model: str
    sft_jsonl: str
    experiment: str
    run_name: str
    out_dir: str
    lr: float
    batch: int
    epochs: int
    max_seq_length: int
    gradient_accumulation: int
    logging_steps: int
    save_steps: int
    quantization: str


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    return TrainConfig(
        base_model=cfg.get("base_model", ""),
        sft_jsonl=data.get("sft_jsonl", ""),
        experiment=training.get("experiment", "llm-finetune"),
        run_name=training.get("run_name", "lora-run"),
        out_dir=training.get("out_dir", "./artifacts/lora_out"),
        lr=float(training.get("lr", 2.0e-4)),
        batch=int(training.get("batch", 1)),
        epochs=int(training.get("epochs", 1)),
        max_seq_length=int(training.get("max_seq_length", 1024)),
        gradient_accumulation=int(training.get("gradient_accumulation", 1)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 200)),
        quantization=str(training.get("quantization", "none")),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return p.parse_args()


def make_model_and_tokenizer(base_model: str, quantization: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if quantization == "4bit":
        load_kwargs = dict(load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
    elif quantization == "8bit":
        load_kwargs = dict(load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
    else:
        load_kwargs = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer, lora_cfg


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    tcfg = build_train_config(cfg)

    if not tcfg.base_model:
        raise SystemExit("base_model nao informado no config.")
    if not tcfg.sft_jsonl or not os.path.exists(tcfg.sft_jsonl):
        raise SystemExit("sft_jsonl nao encontrado: " + str(tcfg.sft_jsonl))

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(tcfg.experiment)

    dataset = load_dataset("json", data_files=tcfg.sft_jsonl, split="train")

    model, tokenizer, lora_cfg = make_model_and_tokenizer(tcfg.base_model, tcfg.quantization)

    training_args = TrainingArguments(
        output_dir=tcfg.out_dir,
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.batch,
        gradient_accumulation_steps=tcfg.gradient_accumulation,
        learning_rate=tcfg.lr,
        logging_steps=tcfg.logging_steps,
        save_steps=tcfg.save_steps,
        fp16=torch.cuda.is_available() and tcfg.quantization == "none",
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=tcfg.max_seq_length,
        args=training_args,
        packing=False,
    )

    with mlflow.start_run(run_name=tcfg.run_name) as run:
        mlflow.log_param("base_model", tcfg.base_model)
        mlflow.log_param("sft_jsonl", tcfg.sft_jsonl)
        mlflow.log_param("lr", tcfg.lr)
        mlflow.log_param("batch", tcfg.batch)
        mlflow.log_param("epochs", tcfg.epochs)
        mlflow.log_param("max_seq_length", tcfg.max_seq_length)
        mlflow.log_param("quantization", tcfg.quantization)

        trainer.train()

        local_art_dir = os.path.join(tcfg.out_dir, "lora_artifacts")
        os.makedirs(local_art_dir, exist_ok=True)
        model.save_pretrained(local_art_dir)
        tokenizer.save_pretrained(local_art_dir)

        manifest = {
            "base_model": tcfg.base_model,
            "sft_jsonl": tcfg.sft_jsonl,
            "lora_config": lora_cfg.to_dict(),
        }
        with open(os.path.join(local_art_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        mlflow.log_artifacts(local_art_dir, artifact_path="lora_artifacts")
        print("OK: treino concluido. run_id =", run.info.run_id)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pathlib
from typing import List, Dict

DEFAULT_INSTRUCTION = (
    "Você é um assistente. Use o conteúdo abaixo para responder com clareza, "
    "criando explicações e exemplos quando fizer sentido."
)

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunks.append(text[i:j])
        i = max(i + chunk_size - overlap, j)
        if i >= len(text):
            break
    return chunks

def make_sft_records(chunks: List[str], instruction: str) -> List[Dict]:
    # Formato "text" pronto pro TRL SFTTrainer (Instruction/Input/Response)
    records = []
    for c in chunks:
        records.append({
            "text": (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{c}\n\n"
                f"### Response:\n"
            )
        })
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--chunk_size", type=int, default=1500)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    args = ap.parse_args()

    text = pathlib.Path(args.input_txt).read_text(encoding="utf-8")
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise SystemExit("Documento vazio: nada para gerar SFT.")

    records = make_sft_records(chunks, args.instruction)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"OK: gerado {len(records)} exemplos em {args.output_jsonl}")

if __name__ == "__main__":
    main()
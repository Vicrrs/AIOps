import argparse
import json
import os
import pathlib
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
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

def build_index(chunks: List[str]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=200_000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(chunks)
    X = normalize(X)
    return vectorizer, X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", required=True)
    ap.add_argument("--rag_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    args = ap.parse_args()

    text = pathlib.Path(args.input_txt).read_text(encoding="utf-8")
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise SystemExit("Documento vazio: nada para indexar.")

    vectorizer, X = build_index(chunks)

    os.makedirs(args.rag_dir, exist_ok=True)
    # artefatos do índice
    joblib.dump(vectorizer, os.path.join(args.rag_dir, "vectorizer.joblib"))
    joblib.dump(X, os.path.join(args.rag_dir, "tfidf_matrix.joblib"))
    with open(os.path.join(args.rag_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, "text": c}, ensure_ascii=False) + "\n")

    meta = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "n_chunks": len(chunks),
        "type": "tfidf_rag",
    }
    with open(os.path.join(args.rag_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"OK: RAG index criado em {args.rag_dir} com {len(chunks)} chunks")

if __name__ == "__main__":
    main()
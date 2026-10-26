"""Batch runner for RAG evaluation."""
import os
import csv
import argparse
from typing import List

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from .config import (
    EMBEDDING_MODELS, LLM_MODELS, DEFAULT_QUESTIONS,
    OUTPUT_DIR, FAISS_INDEX_DIR,
)
from .index import build_site_index, save_site_index, load_site_index, list_available_sites
from .retrieve import retrieve
from .generate import generate
from .verify import verify_answer


def resolve_snapshot(model_path: str) -> str:
    """If path ends in /snapshots, pick the first snapshot hash."""
    if model_path.endswith("/snapshots"):
        snaps = os.listdir(model_path)
        if snaps:
            return os.path.join(model_path, snaps[0])
    return model_path


def get_embeddings(
    model_key: str = "e5-mistral-7b",
    device: str | None = None,
    batch_size: int | None = None,
):
    """Load embedding model with configurable device and batch size.

    Args:
        model_key: Key in EMBEDDING_MODELS
        device: Embedding device, e.g. "cpu" or "cuda".
            If None, uses RAG_EMBED_DEVICE env var, then falls back to
            "cuda" when available else "cpu".
        batch_size: Embedding batch size. If None, uses RAG_EMBED_BATCH_SIZE
            env var, then defaults to 8.
    """
    path = resolve_snapshot(EMBEDDING_MODELS[model_key])

    if device is None:
        device = os.environ.get("RAG_EMBED_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if batch_size is None:
        batch_size_env = os.environ.get("RAG_EMBED_BATCH_SIZE")
        if batch_size_env and batch_size_env.isdigit():
            batch_size = int(batch_size_env)
        else:
            batch_size = 8

    return HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )


def get_or_build_index(site_key: str, embeddings):
    """Load index from disk or build if not exists."""
    index = load_site_index(site_key, embeddings)
    if index is None:
        print(f"  Building index for {site_key}...")
        index = build_site_index(site_key, embeddings)
        if index:
            save_site_index(index, site_key)
    return index


def run_batch(
    site_keys: List[str],
    questions: List[str],
    llm_models: List[str],
    embed_model: str = "e5-mistral-7b",
    output_file: str = None,
    query_expansion: bool = False,
    rerank: bool = False,
):
    """Run batch evaluation across sites, questions, and models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "rag_results.csv")

    embeddings = get_embeddings(embed_model)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "llm_model", "site_key", "question_id", "question",
            "answer", "evidence_quote", "evidence_source",
            "citation_verified"
        ])
        writer.writeheader()

        for model_key in llm_models:
            if model_key not in LLM_MODELS:
                print(f"Unknown model '{model_key}', skipping.")
                continue

            model_path = resolve_snapshot(LLM_MODELS[model_key])
            print(f"\nLoading LLM: {model_key}")
            pipe = pipeline(
                "text-generation",
                model=model_path,
                max_new_tokens=512,
                do_sample=False,
                device_map="auto",
                trust_remote_code=True,
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            for site_key in site_keys:
                index = get_or_build_index(site_key, embeddings)
                if index is None:
                    print(f"  No index for {site_key}, skipping.")
                    continue

                for q_id, question in enumerate(questions):
                    print(f"  [{site_key}] Q{q_id}: {question[:50]}...")

                    chunks = retrieve(index, question, query_expansion=query_expansion, rerank=rerank)
                    answer = generate(llm, question, chunks, site_key)
                    verification = verify_answer(answer, chunks)

                    writer.writerow({
                        "llm_model": model_key,
                        "site_key": site_key,
                        "question_id": q_id,
                        "question": question,
                        "answer": answer.text,
                        "evidence_quote": answer.evidence_quote,
                        "evidence_source": answer.evidence_source,
                        "citation_verified": verification.is_valid,
                    })
                    f.flush()
                    print(f"    A: {answer.text[:80]}...")

            del pipe, llm
            torch.cuda.empty_cache()

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="RAG batch evaluation")
    parser.add_argument("--sites", nargs="+", help="Site keys to process (default: all)")
    parser.add_argument("--questions_file", help="Questions file (one per line)")
    parser.add_argument("--llm_models", nargs="+", default=["qwen3-8b"])
    parser.add_argument("--embed_model", default="e5-mistral-7b")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--query-expansion", action="store_true", help="Enable query expansion for historical/causal questions")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    args = parser.parse_args()

    # Get sites
    sites = args.sites or list_available_sites()[:5]  # Default: first 5 sites

    # Get questions
    questions = DEFAULT_QUESTIONS
    if args.questions_file and os.path.exists(args.questions_file):
        with open(args.questions_file) as f:
            questions = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    run_batch(
        site_keys=sites,
        questions=questions,
        llm_models=args.llm_models,
        embed_model=args.embed_model,
        output_file=args.output,
        query_expansion=args.query_expansion,
        rerank=args.rerank,
    )


if __name__ == "__main__":
    main()

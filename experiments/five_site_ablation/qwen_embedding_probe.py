#!/usr/bin/env python3
"""Verify that raw and instructed Qwen3 queries produce distinct vectors.

This is a GPU integration probe, not a benchmark arm.  It loads the exact
Qwen3 embedding snapshot once, encodes each canonical question both raw and
with the frozen Superfund instruction, and emits JSON Lines suitable for a
run log or archival file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Iterable, Sequence

from embedding_modes import (
    QueryAwareEmbeddings,
    build_embedding_profile,
    validate_embedding_model_assets,
)
from question_io import load_questions


DEFAULT_QWEN_MODEL_PATH = (
    "/work/pi_larsonj_wit_edu/superfund/project/superfund/models/"
    "qwen3-embedding-8b-1d8ad4ca-complete"
)


def vector_sha256(values: Sequence[float]) -> str:
    payload = b"".join(struct.pack("!d", float(value)) for value in values)
    return hashlib.sha256(payload).hexdigest()


def vector_comparison(raw: Sequence[float], instructed: Sequence[float]) -> dict:
    if len(raw) != len(instructed) or not raw:
        raise ValueError("vectors must be non-empty and have equal dimensions")
    raw_norm = math.sqrt(sum(float(value) ** 2 for value in raw))
    instructed_norm = math.sqrt(sum(float(value) ** 2 for value in instructed))
    if raw_norm == 0.0 or instructed_norm == 0.0:
        raise ValueError("vectors must have non-zero norms")
    cosine = sum(float(a) * float(b) for a, b in zip(raw, instructed)) / (
        raw_norm * instructed_norm
    )
    max_abs_delta = max(abs(float(a) - float(b)) for a, b in zip(raw, instructed))
    return {
        "dimension": len(raw),
        "cosine_similarity": cosine,
        "max_abs_delta": max_abs_delta,
        "exact_equal": all(float(a) == float(b) for a, b in zip(raw, instructed)),
        "allclose_atol_1e_6": max_abs_delta <= 1e-6,
        "raw_sha256": vector_sha256(raw),
        "instructed_sha256": vector_sha256(instructed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions_mining.tsv"),
    )
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from langchain_huggingface import HuggingFaceEmbeddings

    profile = build_embedding_profile("qwen3-embed-8b", "superfund")
    asset_manifest = validate_embedding_model_assets("qwen3-embed-8b", args.model_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=args.model_path,
        model_kwargs={"device": args.device},
        encode_kwargs={"batch_size": args.batch_size, "normalize_embeddings": True},
    )
    adapter = QueryAwareEmbeddings(embeddings, profile)
    comparisons = []
    for question_id, question in load_questions(args.questions):
        raw = embeddings.embed_query(question)
        instructed = embeddings.embed_query(profile.transform_query(question))
        wrapped = adapter.embed_query(question)
        if vector_sha256(instructed) != vector_sha256(wrapped):
            raise RuntimeError("query-aware adapter differs from direct instructed encoding")
        result = {
            "record_type": "question",
            "question_id": question_id,
            "question": question,
            **vector_comparison(raw, instructed),
        }
        comparisons.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "record_type": "summary",
                "model_path": str(args.model_path),
                "model_asset_bundle_sha256": asset_manifest["model_asset_bundle_sha256"],
                "pooling_mode": asset_manifest["pooling_mode"],
                "query_instruction_mode": profile.query_mode,
                "query_instruction_sha256": profile.instruction_sha256,
                "num_questions": len(comparisons),
                "num_exact_equal": sum(row["exact_equal"] for row in comparisons),
                "num_allclose_atol_1e_6": sum(
                    row["allclose_atol_1e_6"] for row in comparisons
                ),
                "min_cosine_similarity": min(
                    row["cosine_similarity"] for row in comparisons
                ),
                "max_cosine_similarity": max(
                    row["cosine_similarity"] for row in comparisons
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

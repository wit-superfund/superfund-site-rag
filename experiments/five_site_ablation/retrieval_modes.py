"""Integration adapter for dense, BM25, and hybrid benchmark retrieval.

Copy this file and ``bm25_rrf.py`` beside the cached ``compare_rag.py``.  The
benchmark owns chunk creation and the FAISS cache; this module owns the lexical
cache and query-time selection/fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from bm25_rrf import (
    BM25Config,
    BM25Index,
    bm25_cache_config,
    bm25_retrieve,
    reciprocal_rank_fusion,
    retrieval_config,
)
from rag_cache import CacheEvent, get_or_build_directory


RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "dense"
    candidate_k: int = 30
    top_k: int = 6
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_tokenizer: str = "unicode-alnum-2-64-v1"
    rrf_k: int = 60

    @property
    def bm25_config(self) -> BM25Config:
        return BM25Config(
            k1=self.bm25_k1,
            b=self.bm25_b,
            tokenizer_policy=self.bm25_tokenizer,
        )

    def as_dict(self) -> dict[str, Any]:
        return retrieval_config(
            self.mode,
            candidate_k=self.candidate_k,
            top_k=self.top_k,
            rrf_k=self.rrf_k,
            bm25_config=self.bm25_config,
        )

    @property
    def needs_dense(self) -> bool:
        return self.mode in {"dense", "hybrid"}

    @property
    def needs_bm25(self) -> bool:
        return self.mode in {"bm25", "hybrid"}

    @property
    def score_semantics(self) -> str:
        if self.mode == "dense":
            return "faiss-distance-lower-is-better"
        if self.mode == "bm25":
            return "bm25-score-higher-is-better"
        return "rrf-score-higher-is-better"


def get_or_build_bm25(
    cache_dir: str,
    chunk_key: str,
    chunks: Sequence[Any],
    config: BM25Config,
) -> tuple[BM25Index, CacheEvent]:
    """Load/build a lexical index keyed by exact chunks and BM25 settings."""
    cache_config = bm25_cache_config(chunk_key, config)

    def load(path: Path) -> BM25Index:
        index = BM25Index.load(path)
        if index.document_count != len(chunks):
            raise ValueError("BM25 cache and chunk cache document counts differ")
        return index

    def build(path: Path) -> tuple[BM25Index, dict[str, Any]]:
        index = BM25Index.build((chunk.page_content for chunk in chunks), config)
        index.save(path)
        posting_pairs = sum(len(doc_ids) for doc_ids, _ in index.postings.values())
        return index, {
            "num_documents": index.document_count,
            "vocabulary_size": len(index.postings),
            "posting_pairs": posting_pairs,
            "average_document_length": index.average_document_length,
        }

    return get_or_build_directory(cache_dir, "bm25", cache_config, load, build)


def retrieve_question(
    settings: RetrievalSettings,
    *,
    question: str,
    chunks: Sequence[Any],
    retrieved_chunk_type: type,
    dense_index: Any = None,
    bm25_index: BM25Index | None = None,
    dense_retrieve: Callable[..., list[Any]] | None = None,
) -> list[Any]:
    """Run the selected retrieval arm and return RetrievedChunk-compatible objects."""
    # Force validation even when the caller did not serialize the settings.
    settings.as_dict()
    if settings.needs_dense and (dense_index is None or dense_retrieve is None):
        raise ValueError(f"{settings.mode} retrieval requires a dense index and retrieve function")
    if settings.needs_bm25 and bm25_index is None:
        raise ValueError(f"{settings.mode} retrieval requires a BM25 index")

    if settings.mode == "dense":
        return dense_retrieve(dense_index, question, top_k=settings.top_k)

    if settings.mode == "bm25":
        candidates = bm25_retrieve(
            bm25_index,
            chunks,
            question,
            top_k=settings.candidate_k,
            chunk_type=retrieved_chunk_type,
        )
        return candidates[:settings.top_k]

    dense_candidates = dense_retrieve(dense_index, question, top_k=settings.candidate_k)
    bm25_candidates = bm25_retrieve(
        bm25_index,
        chunks,
        question,
        top_k=settings.candidate_k,
        chunk_type=retrieved_chunk_type,
    )
    # Dense is intentionally the first list and therefore wins exact RRF ties.
    return reciprocal_rank_fusion(
        [dense_candidates, bm25_candidates],
        top_k=settings.top_k,
        rrf_k=settings.rrf_k,
    )

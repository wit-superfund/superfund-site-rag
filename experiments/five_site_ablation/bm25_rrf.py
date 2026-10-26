"""Dependency-free BM25 and reciprocal-rank fusion for Superfund retrieval.

The implementation is designed for the site's 100k+ chunk corpora:

* the index is inverted, so a query visits postings for query terms rather
  than scanning every document;
* posting document IDs and term frequencies use compact ``array('I')``
  storage (eight payload bytes per term-document pair);
* top-k selection uses a bounded heap and deterministic document-ID ties;
* the persisted artifact contains no source text.  Text and metadata continue
  to come from the content-addressed chunk cache.

Only self-generated cache files should be loaded: persistence uses pickle for
compact, fast serialization of the posting arrays.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import pickle
import re
from array import array
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BM25_INDEX_FORMAT_VERSION = 1
BM25_SCHEMA = "superfund-bm25-v1"
RETRIEVAL_SCHEMA = "superfund-retrieval-v1"
TOKENIZER_UNICODE_ALNUM_V1 = "unicode-alnum-2-64-v1"
TOKENIZER_POLICIES = (TOKENIZER_UNICODE_ALNUM_V1,)

# Unicode letters and digits, excluding underscore.  The length limits remove
# one-character noise and pathological OCR runs while retaining EPA IDs.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, policy: str = TOKENIZER_UNICODE_ALNUM_V1) -> list[str]:
    """Tokenize text according to a named, versioned policy."""
    if policy != TOKENIZER_UNICODE_ALNUM_V1:
        raise ValueError(f"unsupported tokenizer policy: {policy!r}")
    return [token for token in _TOKEN_RE.findall(text.casefold()) if 2 <= len(token) <= 64]


@dataclass(frozen=True)
class BM25Config:
    """Versioned BM25 settings included in the lexical cache key."""

    k1: float = 1.2
    b: float = 0.75
    tokenizer_policy: str = TOKENIZER_UNICODE_ALNUM_V1

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be finite and between 0 and 1")
        if self.tokenizer_policy not in TOKENIZER_POLICIES:
            raise ValueError(f"unsupported tokenizer policy: {self.tokenizer_policy!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "tokenizer_policy": self.tokenizer_policy,
        }


@dataclass(frozen=True)
class BM25Hit:
    document_index: int
    score: float


@dataclass
class BM25Index:
    """Compact inverted BM25 index.

    ``postings`` maps a term to parallel unsigned-int arrays of document IDs
    and within-document term frequencies.  Document IDs are assigned in chunk
    cache order and therefore also provide the final deterministic tie-break.
    """

    config: BM25Config
    document_count: int
    average_document_length: float
    document_lengths: array
    postings: dict[str, tuple[array, array]]

    @classmethod
    def build(cls, texts: Iterable[str], config: BM25Config | None = None) -> "BM25Index":
        config = config or BM25Config()
        lengths = array("I")
        postings: dict[str, tuple[array, array]] = {}

        for document_index, text in enumerate(texts):
            if document_index > 0xFFFFFFFF:
                raise OverflowError("BM25 index supports at most 2^32 documents")
            counts = Counter(tokenize(text, config.tokenizer_policy))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                if frequency > 0xFFFFFFFF:
                    raise OverflowError("term frequency exceeds uint32 storage")
                doc_ids, frequencies = postings.setdefault(term, (array("I"), array("I")))
                doc_ids.append(document_index)
                frequencies.append(frequency)

        document_count = len(lengths)
        average_length = (sum(lengths) / document_count) if document_count else 0.0
        return cls(config, document_count, average_length, lengths, postings)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        """Return descending BM25 results; ties use ascending document ID."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.document_count or not self.average_document_length:
            return []

        # Repeated query words must not silently increase document scores.
        # Sorting avoids hash-seed-dependent floating-point accumulation order.
        query_terms = sorted(set(tokenize(query, self.config.tokenizer_policy)))
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        k1 = self.config.k1
        b = self.config.b
        avgdl = self.average_document_length
        count = self.document_count

        for term in query_terms:
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_ids, frequencies = posting
            document_frequency = len(doc_ids)
            # Robertson/Sparck Jones IDF with the commonly used +1 guard.
            idf = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_index, frequency in zip(doc_ids, frequencies):
                length = self.document_lengths[document_index]
                denominator = frequency + k1 * (1.0 - b + b * length / avgdl)
                contribution = idf * (frequency * (k1 + 1.0)) / denominator
                scores[document_index] = scores.get(document_index, 0.0) + contribution

        # nsmallest over (-score, doc_id) avoids sorting all matching documents.
        best = heapq.nsmallest(top_k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [BM25Hit(document_index, score) for document_index, score in best]

    def save(self, directory: str | os.PathLike[str]) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": BM25_INDEX_FORMAT_VERSION,
            "config": self.config.as_dict(),
            "document_count": self.document_count,
            "average_document_length": self.average_document_length,
            "document_lengths": self.document_lengths,
            "postings": self.postings,
        }
        path = destination / "bm25.pkl"
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> "BM25Index":
        with open(Path(directory) / "bm25.pkl", "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format_version") != BM25_INDEX_FORMAT_VERSION:
            raise ValueError("unsupported BM25 index format")
        config = BM25Config(**payload["config"])
        value = cls(
            config=config,
            document_count=payload["document_count"],
            average_document_length=payload["average_document_length"],
            document_lengths=payload["document_lengths"],
            postings=payload["postings"],
        )
        if len(value.document_lengths) != value.document_count:
            raise ValueError("BM25 document length count mismatch")
        return value


def bm25_cache_config(chunk_key: str, config: BM25Config) -> dict[str, Any]:
    """Configuration for the content-addressed lexical index artifact."""
    return {
        "schema": BM25_SCHEMA,
        "format_version": BM25_INDEX_FORMAT_VERSION,
        "chunk_key": chunk_key,
        **config.as_dict(),
    }


def retrieval_config(
    mode: str,
    *,
    candidate_k: int,
    top_k: int,
    rrf_k: int,
    bm25_config: BM25Config,
) -> dict[str, Any]:
    """Materialize all ranking settings for manifests and run fingerprints."""
    if mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"unsupported retrieval mode: {mode!r}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    return {
        "schema": RETRIEVAL_SCHEMA,
        "mode": mode,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "bm25": bm25_config.as_dict(),
        "fusion": "reciprocal-rank-fusion" if mode == "hybrid" else "none",
        "rrf_k": rrf_k,
        "tie_policy": "score-desc,best-rank,first-list,identity",
    }


def _metadata(document: Any) -> Mapping[str, Any]:
    return getattr(document, "metadata", {}) or {}


def _text(document: Any) -> str:
    return getattr(document, "page_content", getattr(document, "text", ""))


def document_to_retrieved_chunk(document: Any, score: float, chunk_type: type) -> Any:
    """Convert a cached LangChain Document to the live RetrievedChunk shape."""
    metadata = _metadata(document)
    return chunk_type(
        text=_text(document),
        pdf_filename=metadata.get("pdf_filename", ""),
        page=metadata.get("page", 0),
        epa_url=metadata.get("epa_url", ""),
        score=score,
    )


def bm25_retrieve(
    index: BM25Index,
    documents: Sequence[Any],
    question: str,
    *,
    top_k: int,
    chunk_type: type,
) -> list[Any]:
    if len(documents) != index.document_count:
        raise ValueError("BM25 index and chunk cache have different document counts")
    return [
        document_to_retrieved_chunk(documents[hit.document_index], hit.score, chunk_type)
        for hit in index.search(question, top_k)
    ]


def chunk_identity(chunk: Any) -> tuple[str, int, str, str]:
    """Stable identity for deduplicating the same chunk across retrievers."""
    text_digest = hashlib.sha256(getattr(chunk, "text", "").encode("utf-8")).hexdigest()
    return (
        getattr(chunk, "pdf_filename", ""),
        int(getattr(chunk, "page", 0)),
        getattr(chunk, "epa_url", ""),
        text_digest,
    )


def _with_score(chunk: Any, score: float) -> Any:
    try:
        return replace(chunk, score=score)
    except (TypeError, ValueError):
        return type(chunk)(
            text=chunk.text,
            pdf_filename=chunk.pdf_filename,
            page=chunk.page,
            epa_url=chunk.epa_url,
            score=score,
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[Any]:
    """Fuse ranked lists with deterministic RRF and exact-chunk deduplication.

    Rank is one-based.  A duplicate within one input list contributes only at
    its first occurrence.  Ties prefer the best individual rank, then the
    earlier input list, then stable source identity.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    fused: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for list_index, ranking in enumerate(rankings):
        seen_in_list: set[tuple[str, int, str, str]] = set()
        for rank, chunk in enumerate(ranking, start=1):
            identity = chunk_identity(chunk)
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            item = fused.setdefault(identity, {
                "chunk": chunk,
                "score": 0.0,
                "best_rank": rank,
                "first_list": list_index,
            })
            item["score"] += 1.0 / (rrf_k + rank)
            item["best_rank"] = min(item["best_rank"], rank)
            item["first_list"] = min(item["first_list"], list_index)

    ordered = sorted(
        fused.items(),
        key=lambda pair: (
            -pair[1]["score"],
            pair[1]["best_rank"],
            pair[1]["first_list"],
            pair[0],
        ),
    )
    return [_with_score(item["chunk"], item["score"]) for _, item in ordered[:top_k]]

"""Retrieval from per-site indexes."""
from dataclasses import dataclass
from typing import List, Optional
from langchain_community.vectorstores import FAISS

from .config import TOP_K

# Cross-encoder for reranking
RERANK_MODEL = None  # Lazy load

# Query expansion patterns for different question types
EXPANSION_PATTERNS = {
    "historical": {
        "triggers": ["earliest", "history", "first", "origin", "when did", "began", "started"],
        "expansion": "site history background origin construction industrial activity timeline"
    },
    "causal": {
        "triggers": ["cause", "source of", "why", "how did", "responsible", "led to"],
        "expansion": "cause source responsible company industrial dumping disposal waste"
    },
    "regulatory": {
        "triggers": ["regulatory status", "npl", "listed", "delisted", "superfund status"],
        "expansion": "NPL National Priorities List status listing delisting regulatory"
    },
}


def expand_query(question: str) -> str:
    """Expand query based on question type to improve retrieval."""
    q_lower = question.lower()
    expansions = []

    for pattern in EXPANSION_PATTERNS.values():
        if any(trigger in q_lower for trigger in pattern["triggers"]):
            expansions.append(pattern["expansion"])

    if expansions:
        return f"{question} {' '.join(expansions)}"
    return question


@dataclass
class RetrievedChunk:
    """A retrieved chunk with source information."""
    text: str
    pdf_filename: str
    page: int
    epa_url: str
    score: float


def get_reranker():
    """Lazy load cross-encoder reranking model."""
    global RERANK_MODEL
    if RERANK_MODEL is None:
        from sentence_transformers import CrossEncoder
        RERANK_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return RERANK_MODEL


def rerank_chunks(question: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
    """Rerank chunks using cross-encoder for better semantic relevance."""
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [(question, c.text) for c in chunks]
    scores = reranker.predict(pairs)

    # Sort by rerank score (higher = more relevant)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def retrieve(index: FAISS, question: str, top_k: int = TOP_K, query_expansion: bool = False, rerank: bool = False) -> List[RetrievedChunk]:
    """Retrieve top-k chunks from a site index.

    Args:
        index: FAISS vectorstore for a single site
        question: Query string
        top_k: Number of chunks to retrieve
        query_expansion: If True, expand query for better historical/causal retrieval
        rerank: If True, rerank results with cross-encoder

    Returns: List of RetrievedChunk with source info
    """
    query = expand_query(question) if query_expansion else question
    # Fetch more candidates if reranking
    fetch_k = top_k * 3 if rerank else top_k
    results = index.similarity_search_with_score(query, k=fetch_k)

    chunks = []
    for doc, score in results:
        chunks.append(RetrievedChunk(
            text=doc.page_content,
            pdf_filename=doc.metadata.get("pdf_filename", ""),
            page=doc.metadata.get("page", 0),
            epa_url=doc.metadata.get("epa_url", ""),
            score=score,
        ))

    if rerank:
        chunks = rerank_chunks(question, chunks, top_k)

    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as context string with source markers."""
    parts = []
    for i, c in enumerate(chunks):
        source = f"[{c.pdf_filename}, p.{c.page}]"
        parts.append(f"Source {i+1} {source}:\n{c.text}")
    return "\n\n".join(parts)


def get_evidence_summary(chunks: List[RetrievedChunk], max_quotes: int = 3) -> tuple:
    """Get evidence quote and source for CSV output.

    Returns: (evidence_quote, evidence_source)
    """
    if not chunks:
        return "", ""

    quotes = []
    sources = []
    for c in chunks[:max_quotes]:
        # Take first 200 chars as quote
        quote = c.text[:200].replace("\n", " ").strip()
        if len(c.text) > 200:
            quote += "..."
        quotes.append(quote)
        sources.append(f"{c.pdf_filename}:p{c.page}")

    return " | ".join(quotes), " | ".join(sources)

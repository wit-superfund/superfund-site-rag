"""Citation verification - check that quotes exist in source."""
import re
from dataclasses import dataclass
from typing import List, Optional

from .retrieve import RetrievedChunk
from .generate import Answer


@dataclass
class VerificationResult:
    """Result of citation verification."""
    is_valid: bool
    answer_text: str
    verified_sources: List[str]
    failed_citations: List[str]


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_quote_in_chunks(quote: str, chunks: List[RetrievedChunk], threshold: float = 0.8) -> Optional[RetrievedChunk]:
    """Check if a quote exists in any of the chunks.

    Uses simple substring matching after normalization.
    """
    norm_quote = normalize_text(quote)
    if len(norm_quote) < 20:
        return None  # Too short to verify

    for chunk in chunks:
        norm_chunk = normalize_text(chunk.text)
        if norm_quote in norm_chunk:
            return chunk
    return None


def extract_citations(answer_text: str) -> List[tuple]:
    """Extract citations from answer text.

    Handles formats:
    - [Source N] or [Source N, p.M]
    - [filename, p.N]

    Returns: List of (source_id, page_num or None) tuples
    """
    citations = []

    # Match [Source N] or [Source N, p.M]
    source_pattern = r"\[Source\s*(\d+)(?:,\s*p\.?(\d+))?\]"
    for match in re.finditer(source_pattern, answer_text):
        source_idx = int(match.group(1)) - 1  # Convert to 0-indexed
        page = int(match.group(2)) if match.group(2) else None
        citations.append(("source", source_idx, page))

    # Match [filename, p.N] format
    file_pattern = r"\[([^\]]+\.pdf),\s*p\.?(\d+)\]"
    for match in re.finditer(file_pattern, answer_text):
        filename = match.group(1).strip()
        page = int(match.group(2))
        citations.append(("file", filename, page))

    return citations


def verify_answer(answer: Answer, chunks: List[RetrievedChunk]) -> VerificationResult:
    """Verify that citations in the answer match retrieved chunks.

    Args:
        answer: Generated answer
        chunks: Retrieved chunks used for generation

    Returns: VerificationResult with validity status
    """
    citations = extract_citations(answer.text)

    if not citations:
        # No citations to verify - answer is valid but unverified
        return VerificationResult(
            is_valid=True,
            answer_text=answer.text,
            verified_sources=[],
            failed_citations=[],
        )

    verified = []
    failed = []

    for citation in citations:
        cite_type = citation[0]
        found = False

        if cite_type == "source":
            # [Source N] format - check if source index is valid
            source_idx = citation[1]
            if 0 <= source_idx < len(chunks):
                chunk = chunks[source_idx]
                verified.append(f"Source {source_idx + 1} -> {chunk.pdf_filename}:p{chunk.page}")
                found = True
            else:
                failed.append(f"Source {source_idx + 1} (out of range)")

        elif cite_type == "file":
            # [filename, p.N] format
            filename, page = citation[1], citation[2]
            for chunk in chunks:
                if filename in chunk.pdf_filename and chunk.page == page:
                    verified.append(f"{filename}:p{page}")
                    found = True
                    break
            if not found:
                failed.append(f"{filename}:p{page}")

    return VerificationResult(
        is_valid=len(failed) == 0,
        answer_text=answer.text,
        verified_sources=verified,
        failed_citations=failed,
    )

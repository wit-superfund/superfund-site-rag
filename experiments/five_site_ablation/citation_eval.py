"""Citation-contract evaluation for Superfund RAG answers.

This module deliberately evaluates a narrow claim: whether the citations in an
answer point to retrieved evidence. It does *not* determine whether every claim
is entailed by that evidence. Keeping those concepts separate prevents a valid
source index from being reported as proof that an answer is hallucination-free.

Two modes support a staged rollout:

``lenient``
    Backward-compatible answer validity. A substantive answer with no parsed
    citations remains valid-but-unverified, matching the historical verifier.
    Grouped citations are nevertheless parsed and checked correctly.

``strict``
    Fail closed for substantive answers. At least one well-formed citation is
    required and every citation must resolve to retrieved evidence. The exact
    ``NOT PRESENT`` sentinel is a valid abstention for which citation
    verification is not applicable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple


EvaluationMode = Literal["lenient", "strict"]


@dataclass(frozen=True)
class Citation:
    """One citation parsed from an answer."""

    kind: Literal["source", "file"]
    raw: str
    source_number: Optional[int] = None
    filename: Optional[str] = None
    page: Optional[int] = None

    @property
    def label(self) -> str:
        if self.kind == "source":
            suffix = f", p.{self.page}" if self.page is not None else ""
            return f"Source {self.source_number}{suffix}"
        return f"{self.filename}:p{self.page}"


@dataclass(frozen=True)
class CitationParseResult:
    citations: Tuple[Citation, ...]
    issues: Tuple[str, ...] = ()


@dataclass
class VerificationResult:
    """Backward-compatible result plus explicit evaluation semantics.

    The first four fields retain the interface consumed by the existing Unity
    runners. New code should use ``status``, ``citation_verified`` and
    ``reason_codes`` rather than treating ``is_valid`` as citation accuracy.
    """

    is_valid: bool
    answer_text: str
    verified_sources: List[str]
    failed_citations: List[str]
    status: str = ""
    citation_verified: Optional[bool] = None
    is_abstention: bool = False
    citation_count: int = 0
    parse_issues: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    semantic_support_checked: bool = False
    mode: EvaluationMode = "lenient"


_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_SOURCE_WORD = re.compile(r"\bsources?\b", re.IGNORECASE)
_PAGE = re.compile(r"\bp(?:age)?\.?\s*(\d+)\b", re.IGNORECASE)
_FILE = re.compile(
    r"^\s*(?P<filename>[^\[\]]+?\.pdf)\s*[,;]?\s*"
    r"p(?:age)?\.?\s*(?P<page>\d+)\s*$",
    re.IGNORECASE,
)
_NUMBER_OR_RANGE = re.compile(r"(?<!\d)(\d+)(?:\s*[-–—]\s*(\d+))?(?!\d)")


def _source_numbers(text: str) -> Tuple[List[int], Optional[str]]:
    """Extract source numbers/ranges after source and page tokens are removed."""

    without_pages = _PAGE.sub(" ", text)
    without_words = _SOURCE_WORD.sub(" ", without_pages)
    numbers: List[int] = []
    chars = list(without_words)
    for match in _NUMBER_OR_RANGE.finditer(without_words):
        start = int(match.group(1))
        if match.group(2) is None:
            numbers.append(start)
        else:
            end = int(match.group(2))
            if end < start or end - start > 99:
                return [], "invalid_source_range"
            numbers.extend(range(start, end + 1))
        left, right = match.span()
        chars[left:right] = " " * (right - left)
    remainder = "".join(chars)

    # After known separators are removed, unexpected words/punctuation signal
    # a source-looking bracket we cannot interpret safely.
    residue = re.sub(r"\b(?:and|or)\b|[,&;/+\s]", "", remainder, flags=re.IGNORECASE)
    if residue:
        return [], "malformed_source_citation"
    if not numbers:
        return [], "malformed_source_citation"
    return numbers, None


def parse_citations(answer_text: str) -> CitationParseResult:
    """Parse source and filename citations, including grouped source forms.

    Supported examples include ``[Source 3]``, ``[Source 3, Source 4]``,
    ``[Sources 3 and 4]``, ``[Source 3; 4]``, ``[Sources 3-5]``,
    ``[Source 3, p.12]`` and ``[report.pdf, p.12]``.
    """

    citations: List[Citation] = []
    issues: List[str] = []

    for bracket in _BRACKET.finditer(answer_text or ""):
        raw = bracket.group(0)
        body = bracket.group(1).strip()

        file_match = _FILE.fullmatch(body)
        if file_match:
            citations.append(
                Citation(
                    kind="file",
                    raw=raw,
                    filename=file_match.group("filename").strip(),
                    page=int(file_match.group("page")),
                )
            )
            continue

        if not _SOURCE_WORD.search(body):
            continue

        numbers, issue = _source_numbers(body)
        if issue:
            issues.append(issue)
            continue

        pages = [int(value) for value in _PAGE.findall(body)]
        if not pages:
            citation_pages: Sequence[Optional[int]] = [None] * len(numbers)
        elif len(numbers) == 1 and len(pages) == 1:
            citation_pages = pages
        elif len(numbers) == len(pages):
            citation_pages = pages
        else:
            issues.append("ambiguous_page_qualifier")
            citation_pages = [None] * len(numbers)

        citations.extend(
            Citation(kind="source", raw=raw, source_number=number, page=page)
            for number, page in zip(numbers, citation_pages)
        )

    # Preserve repeated references. They can represent separate claim-level
    # citation occurrences and should remain visible in citation-count metrics.
    return CitationParseResult(tuple(citations), tuple(dict.fromkeys(issues)))


def extract_citations(answer_text: str) -> List[tuple]:
    """Compatibility adapter for the historical tuple-returning API."""

    parsed = parse_citations(answer_text)
    tuples = []
    for citation in parsed.citations:
        if citation.kind == "source":
            tuples.append(("source", citation.source_number - 1, citation.page))
        else:
            tuples.append(("file", citation.filename, citation.page))
    return tuples


def _chunk_filename(chunk: object) -> str:
    return str(getattr(chunk, "pdf_filename", ""))


def _chunk_page(chunk: object) -> Optional[int]:
    page = getattr(chunk, "page", None)
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _filename_equal(left: str, right: str) -> bool:
    return os.path.basename(left).casefold() == os.path.basename(right).casefold()


def _is_exact_abstention(text: str) -> bool:
    return text.strip() == "NOT PRESENT"


def _contains_abstention_token(text: str) -> bool:
    return "NOT PRESENT" in text.upper()


def verify_text(
    answer_text: str,
    chunks: Sequence[object],
    *,
    mode: EvaluationMode = "lenient",
) -> VerificationResult:
    """Evaluate an answer's citation contract against retrieved chunks."""

    if mode not in ("lenient", "strict"):
        raise ValueError(f"unsupported evaluation mode: {mode}")

    text = answer_text or ""
    parsed = parse_citations(text)

    if _is_exact_abstention(text):
        return VerificationResult(
            is_valid=True,
            answer_text=text,
            verified_sources=[],
            failed_citations=[],
            status="abstained",
            citation_verified=None,
            is_abstention=True,
            citation_count=0,
            parse_issues=list(parsed.issues),
            reason_codes=["abstention_exact", "citation_not_applicable"],
            mode=mode,
        )

    if _contains_abstention_token(text):
        strict_valid = mode == "lenient"
        return VerificationResult(
            is_valid=strict_valid,
            answer_text=text,
            verified_sources=[],
            failed_citations=["NOT PRESENT mixed with substantive content"],
            status="mixed_abstention",
            citation_verified=False,
            is_abstention=False,
            citation_count=len(parsed.citations),
            parse_issues=list(parsed.issues),
            reason_codes=["mixed_abstention"],
            mode=mode,
        )

    verified: List[str] = []
    failed: List[str] = []
    reasons: List[str] = list(parsed.issues)

    for citation in parsed.citations:
        if citation.kind == "source":
            assert citation.source_number is not None
            index = citation.source_number - 1
            if index < 0 or index >= len(chunks):
                failed.append(f"Source {citation.source_number} (out of range)")
                reasons.append("source_out_of_range")
                continue

            chunk = chunks[index]
            actual_page = _chunk_page(chunk)
            if citation.page is not None and actual_page != citation.page:
                failed.append(
                    f"Source {citation.source_number}, p.{citation.page} "
                    f"(retrieved page is {actual_page})"
                )
                reasons.append("source_page_mismatch")
                continue
            verified.append(
                f"Source {citation.source_number} -> "
                f"{_chunk_filename(chunk)}:p{actual_page}"
            )
            continue

        match = next(
            (
                chunk
                for chunk in chunks
                if citation.filename
                and _filename_equal(citation.filename, _chunk_filename(chunk))
                and citation.page == _chunk_page(chunk)
            ),
            None,
        )
        if match is None:
            failed.append(citation.label)
            reasons.append("file_page_not_retrieved")
        else:
            verified.append(citation.label)

    citation_count = len(parsed.citations)
    if citation_count == 0:
        reasons.append("missing_citation")
        if parsed.issues:
            status = "malformed_citation"
        else:
            status = "uncited"
        return VerificationResult(
            is_valid=mode == "lenient",
            answer_text=text,
            verified_sources=verified,
            failed_citations=failed,
            status=status,
            citation_verified=False,
            citation_count=0,
            parse_issues=list(parsed.issues),
            reason_codes=list(dict.fromkeys(reasons)),
            mode=mode,
        )

    if parsed.issues or failed:
        status = "malformed_citation" if parsed.issues else "invalid_citation"
        return VerificationResult(
            is_valid=False,
            answer_text=text,
            verified_sources=verified,
            failed_citations=failed,
            status=status,
            citation_verified=False,
            citation_count=citation_count,
            parse_issues=list(parsed.issues),
            reason_codes=list(dict.fromkeys(reasons)),
            mode=mode,
        )

    return VerificationResult(
        is_valid=True,
        answer_text=text,
        verified_sources=verified,
        failed_citations=[],
        status="verified",
        citation_verified=True,
        citation_count=citation_count,
        parse_issues=[],
        reason_codes=["all_citations_resolve"],
        mode=mode,
    )


def verify_answer(
    answer: object,
    chunks: Sequence[object],
    *,
    mode: EvaluationMode = "lenient",
) -> VerificationResult:
    """Compatibility entry point accepting the existing ``Answer`` object."""

    return verify_text(str(getattr(answer, "text", "")), chunks, mode=mode)

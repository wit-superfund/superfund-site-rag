"""PDF extraction with page-level metadata."""
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from pypdf import PdfReader


@dataclass
class PageChunk:
    """A chunk of text with full provenance."""
    text: str
    site_key: str
    pdf_filename: str
    page: int
    chunk_index: int
    epa_url: Optional[str] = None


def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8).startswith(b"%PDF-")
    except Exception:
        return False


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def extract_pdf_pages(path: str, max_pages: Optional[int] = None) -> List[tuple]:
    """Extract text from PDF, preserving page boundaries.

    Returns: List of (page_num, text) tuples (1-indexed pages).
    """
    if not _looks_like_pdf(path):
        return []
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return []
        pages = []
        selected_pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        for i, page in enumerate(selected_pages):
            try:
                text = _clean(page.extract_text() or "")
            except Exception:
                text = ""
            if text:
                pages.append((i + 1, text))  # 1-indexed
        return pages
    except Exception:
        return []


def extract_site_chunks(
    site_key: str,
    pdf_paths: List[str],
    pdf_urls: dict,
    chunk_size: int = 2400,
    chunk_overlap: int = 300,
    max_pages: Optional[int] = None,
) -> List[PageChunk]:
    """Extract all chunks for a site with full provenance.

    Args:
        site_key: Site identifier
        pdf_paths: List of PDF file paths
        pdf_urls: Dict mapping pdf_filename -> EPA URL
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        max_pages: Optional page cap per PDF; None extracts every page

    Returns: List of PageChunk objects
    """
    chunks = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        epa_url = pdf_urls.get(filename)
        pages = extract_pdf_pages(path, max_pages)

        for page_num, page_text in pages:
            page_chunks = _chunk_text(page_text, chunk_size, chunk_overlap)
            for idx, chunk_text in enumerate(page_chunks):
                if chunk_text.strip():
                    chunks.append(PageChunk(
                        text=chunk_text,
                        site_key=site_key,
                        pdf_filename=filename,
                        page=page_num,
                        chunk_index=idx,
                        epa_url=epa_url,
                    ))
    return chunks

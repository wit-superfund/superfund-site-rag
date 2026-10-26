"""Page-aware PDF extraction with an explicit, auditable fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class PdfExtractionResult:
    pages: list[tuple[int, str]]
    fallback_used: bool = False
    whole_document_fallback: bool = False
    fallback_page_count: int = 0
    failed_page_count: int = 0
    failed_document: bool = False
    primary_error: str = ""
    fallback_error: str = ""


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _page_limit(page_count: int, max_pages: int) -> int:
    return min(page_count, max_pages) if max_pages else page_count


def extract_pdf_pages(
    path: str,
    max_pages: int = 0,
    *,
    pdf_reader_cls: Optional[Callable[[str], Any]] = None,
    pymupdf_module: Any = None,
) -> PdfExtractionResult:
    """Extract pages with pypdf, falling back to PyMuPDF on parser errors.

    Empty pages remain represented so page numbers are never compacted. The
    returned diagnostics distinguish whole-document fallback, page fallback,
    and pages that neither parser could read.
    """
    if pdf_reader_cls is None:
        from pypdf import PdfReader

        pdf_reader_cls = PdfReader

    fallback_doc = None

    def open_fallback():
        nonlocal fallback_doc, pymupdf_module
        if fallback_doc is None:
            if pymupdf_module is None:
                import pymupdf

                pymupdf_module = pymupdf
            fallback_doc = pymupdf_module.open(path)
        return fallback_doc

    try:
        reader = pdf_reader_cls(path)
        page_count = len(reader.pages)
    except Exception as primary_error:
        try:
            doc = open_fallback()
            pages = []
            failed_pages = 0
            for index in range(_page_limit(doc.page_count, max_pages)):
                try:
                    text = (doc[index].get_text() or "").strip()
                except Exception:
                    text = ""
                    failed_pages += 1
                pages.append((index + 1, text))
            return PdfExtractionResult(
                pages=pages,
                fallback_used=True,
                whole_document_fallback=True,
                fallback_page_count=len(pages),
                failed_page_count=failed_pages,
                primary_error=_error_text(primary_error),
            )
        except Exception as fallback_error:
            return PdfExtractionResult(
                pages=[],
                fallback_used=True,
                whole_document_fallback=True,
                failed_document=True,
                primary_error=_error_text(primary_error),
                fallback_error=_error_text(fallback_error),
            )
        finally:
            if fallback_doc is not None:
                fallback_doc.close()

    pages = []
    fallback_count = 0
    failed_count = 0
    first_primary_error = ""
    first_fallback_error = ""
    try:
        for index in range(_page_limit(page_count, max_pages)):
            try:
                text = (reader.pages[index].extract_text() or "").strip()
            except Exception as primary_error:
                if not first_primary_error:
                    first_primary_error = _error_text(primary_error)
                fallback_count += 1
                try:
                    text = (open_fallback()[index].get_text() or "").strip()
                except Exception as fallback_error:
                    if not first_fallback_error:
                        first_fallback_error = _error_text(fallback_error)
                    text = ""
                    failed_count += 1
            pages.append((index + 1, text))
        return PdfExtractionResult(
            pages=pages,
            fallback_used=fallback_count > 0,
            fallback_page_count=fallback_count,
            failed_page_count=failed_count,
            primary_error=first_primary_error,
            fallback_error=first_fallback_error,
        )
    finally:
        if fallback_doc is not None:
            fallback_doc.close()

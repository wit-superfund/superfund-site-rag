"""Per-site FAISS index management."""
import os
import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import SITE_INDEX_DIR, PDFS_DIR, FAISS_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MAX_PAGES_PER_PDF
from .extract import extract_site_chunks, PageChunk


def load_site_json(site_key: str) -> dict:
    """Load site metadata from site_pdf_index/{site_key}.json."""
    path = os.path.join(SITE_INDEX_DIR, f"{site_key}.json")
    with open(path) as f:
        return json.load(f)

def get_site_pdf_paths(site_data: dict) -> tuple:
    """Get PDF paths and URL mapping from site JSON.

    Returns: (pdf_paths, pdf_urls) where pdf_urls maps filename -> EPA URL
    """
    pdf_paths = []
    pdf_urls = {}
    for doc in site_data.get("documents", []):
        filename = doc.get("filename", "")
        path = os.path.join(PDFS_DIR, filename)
        if os.path.exists(path):
            pdf_paths.append(path)
            if doc.get("url"):
                pdf_urls[filename] = doc["url"]
    return pdf_paths, pdf_urls


def chunks_to_documents(chunks: List[PageChunk]) -> List[Document]:
    """Convert PageChunks to LangChain Documents with metadata."""
    docs = []
    for c in chunks:
        docs.append(Document(
            page_content=c.text,
            metadata={
                "site_key": c.site_key,
                "pdf_filename": c.pdf_filename,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "epa_url": c.epa_url or "",
            }
        ))
    return docs


def build_site_index(site_key: str, embeddings) -> Optional[FAISS]:
    """Build FAISS index for a single site.

    Args:
        site_key: Site identifier (matches JSON filename without .json)
        embeddings: LangChain Embeddings instance

    Returns: FAISS vectorstore or None if no documents
    """
    site_data = load_site_json(site_key)
    pdf_paths, pdf_urls = get_site_pdf_paths(site_data)

    if not pdf_paths:
        return None

    chunks = extract_site_chunks(
        site_key=site_key,
        pdf_paths=pdf_paths,
        pdf_urls=pdf_urls,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_pages=MAX_PAGES_PER_PDF,
    )

    if not chunks:
        return None

    docs = chunks_to_documents(chunks)
    return FAISS.from_documents(docs, embeddings)


def save_site_index(index: FAISS, site_key: str):
    """Save site index to disk."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    path = os.path.join(FAISS_INDEX_DIR, site_key)
    index.save_local(path)


def load_site_index(site_key: str, embeddings) -> Optional[FAISS]:
    """Load site index from disk."""
    path = os.path.join(FAISS_INDEX_DIR, site_key)
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None


def list_available_sites() -> List[str]:
    """List all site keys from site_pdf_index/."""
    sites = []
    for f in os.listdir(SITE_INDEX_DIR):
        if f.endswith(".json"):
            sites.append(f[:-5])  # Remove .json
    return sorted(sites)

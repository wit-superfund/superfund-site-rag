"""
Superfund RAG Benchmark — Extraction, Embedding, and LLM comparison.

Experiments:
  1. Extraction: pypdf vs pymupdf vs docling
  2. Embeddings: e5-mistral-7b vs qwen3-embed-8b vs bge-large
  3. LLMs: gemma3-27b vs gemma4-12b vs gemma4-27b

Measures: answer quality, timing per stage, chunk count, index size.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from rag_march6.config import FAISS_INDEX_DIR, SITE_INDEX_DIR, PDFS_DIR
from rag_march6.index import load_site_json, get_site_pdf_paths
from rag_march6.retrieve import retrieve, RetrievedChunk
from rag_march6.generate import build_prompt, strip_cot, Answer
from rag_march6.verify import verify_answer

# ── Config ────────────────────────────────────────────────────────────────────
HF_HUB = "/datasets/ai"

EMBEDDING_MODELS = {
    "e5-mistral-7b": f"{HF_HUB}/intfloat/hub/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1",
    "qwen3-embed-8b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}

LLM_MODELS = {
    "gemma3-27b": f"{HF_HUB}/gemma/hub/models--google--gemma-3-27b-it/snapshots",
    "gemma4-12b": f"{HF_HUB}/gemma/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "gemma4-27b": f"{HF_HUB}/gemma/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445",
}

CHUNK_SIZE = 2400
CHUNK_OVERLAP = 300
MAX_PAGES = None
TOP_K = 6


# ── Extraction Methods ────────────────────────────────────────────────────────
def extract_pypdf(pdf_path: str, max_pages: Optional[int] = MAX_PAGES) -> List[str]:
    """Extract text using pypdf."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(pdf_path)
        pages = []
        selected_pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        for page in selected_pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        return pages
    except Exception as e:
        print(f"    pypdf error on {os.path.basename(pdf_path)}: {e}")
        return []


def extract_pymupdf(pdf_path: str, max_pages: Optional[int] = MAX_PAGES) -> List[str]:
    """Extract text using pymupdf (fitz)."""
    import pymupdf
    try:
        doc = pymupdf.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            text = page.get_text().strip()
            if text:
                pages.append(text)
        doc.close()
        return pages
    except Exception as e:
        print(f"    pymupdf error on {os.path.basename(pdf_path)}: {e}")
        return []


def extract_docling(pdf_path: str, max_pages: Optional[int] = MAX_PAGES) -> List[str]:
    """Extract text using docling with hybrid chunking."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.chunking import HybridChunker
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(pdf_path)
        # Return as page-like chunks from docling's own chunking
        chunker = HybridChunker(max_tokens=512, merge_peers=True)
        chunks = list(chunker.chunk(result.document))
        return [c.text for c in chunks if c.text.strip()]
    except Exception as e:
        print(f"    docling error on {os.path.basename(pdf_path)}: {e}")
        return []


EXTRACTORS = {
    "pypdf": extract_pypdf,
    "pymupdf": extract_pymupdf,
    "docling": extract_docling,
}


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(pages: List[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split pages into overlapping chunks."""
    full_text = "\n\n".join(pages)
    chunks = []
    i = 0
    while i < len(full_text):
        end = min(len(full_text), i + chunk_size)
        chunks.append(full_text[i:end])
        if end == len(full_text):
            break
        i = max(0, end - overlap)
    return chunks


# ── Index Building ────────────────────────────────────────────────────────────
def build_index_for_site(site_key: str, extractor_name: str, embeddings) -> tuple:
    """Build FAISS index for a site using specified extractor.
    Returns (index, timing_dict) or (None, timing_dict).
    """
    timings = {}

    # Get PDFs
    site_data = load_site_json(site_key)
    pdf_paths, pdf_urls = get_site_pdf_paths(site_data)
    if not pdf_paths:
        return None, {"extract_sec": 0, "embed_sec": 0, "num_chunks": 0}

    # Extract
    extractor = EXTRACTORS[extractor_name]
    t0 = time.time()
    all_chunks = []
    for path in pdf_paths:
        if extractor_name == "docling":
            # Docling returns pre-chunked text
            chunks = extractor(path)
        else:
            pages = extractor(path)
            chunks = chunk_text(pages)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata={"site_key": site_key, "pdf_filename": os.path.basename(path), "chunk_index": i}
                ))
    timings["extract_sec"] = time.time() - t0
    timings["num_chunks"] = len(all_chunks)

    if not all_chunks:
        timings["embed_sec"] = 0
        return None, timings

    # Embed + index
    t0 = time.time()
    index = FAISS.from_documents(all_chunks, embeddings)
    timings["embed_sec"] = time.time() - t0

    return index, timings


# ── LLM Loading ───────────────────────────────────────────────────────────────
def resolve_snapshot(model_path: str) -> str:
    if model_path.endswith("/snapshots"):
        snaps = os.listdir(model_path)
        if snaps:
            return os.path.join(model_path, snaps[0])
    return model_path


def load_llm(model_key: str):
    """Load LLM, returns (model, tokenizer, load_time)."""
    model_path = resolve_snapshot(LLM_MODELS[model_key])
    t0 = time.time()

    if "gemma4" in model_key:
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    load_time = time.time() - t0
    return model, tokenizer, load_time


def generate_answer(model, tokenizer, question, chunks, site_key, model_key):
    """Generate answer with timing."""
    if not chunks:
        return Answer(text="No relevant information found.", evidence_quote="", evidence_source="", raw_response=""), 0.0

    prompt_text = build_prompt(question, chunks, site_key)
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen_time = time.time() - t0

    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    text = strip_cot(raw)

    evidence_quotes = []
    evidence_sources = []
    for c in chunks[:3]:
        evidence_quotes.append(c.text.replace("\n", " ").strip()[:200])
        evidence_sources.append(f"{c.pdf_filename}:p{c.page}")

    answer = Answer(
        text=text,
        evidence_quote=" | ".join(evidence_quotes),
        evidence_source=" | ".join(evidence_sources),
        raw_response=raw,
    )
    return answer, gen_time


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark")
    parser.add_argument("--sites", nargs="+", default=["Love_Canal", "Times_Beach", "US_Smelter_and_Lead_Refinery"])
    parser.add_argument("--questions_file", default="questions_ella.txt")
    parser.add_argument("--extractors", nargs="+", default=["pypdf", "pymupdf"], choices=list(EXTRACTORS.keys()))
    parser.add_argument("--embed_models", nargs="+", default=["e5-mistral-7b"], choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--llm_model", default="gemma4-12b", choices=list(LLM_MODELS.keys()))
    parser.add_argument("--output_dir", default="output/benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load questions
    questions = ["What contaminants are present at this site?"]
    if os.path.exists(args.questions_file):
        with open(args.questions_file) as f:
            questions = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Load LLM once
    print(f"Loading LLM: {args.llm_model}")
    model, tokenizer, llm_load_time = load_llm(args.llm_model)
    print(f"  LLM loaded in {llm_load_time:.1f}s")

    # Results
    results_file = os.path.join(args.output_dir, f"benchmark_{args.llm_model}.csv")
    timing_file = os.path.join(args.output_dir, f"timing_{args.llm_model}.csv")

    results_rows = []
    timing_rows = []

    for embed_key in args.embed_models:
        embed_path = EMBEDDING_MODELS[embed_key]
        print(f"\nLoading embeddings: {embed_key}")
        t0 = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"batch_size": 8, "normalize_embeddings": True},
        )
        embed_load_time = time.time() - t0
        print(f"  Embeddings loaded in {embed_load_time:.1f}s")

        for extractor_name in args.extractors:
            print(f"\n  Extractor: {extractor_name} | Embeddings: {embed_key}")

            for site_key in args.sites:
                print(f"    Building index for {site_key}...")
                index, timings = build_index_for_site(site_key, extractor_name, embeddings)

                timing_rows.append({
                    "extractor": extractor_name,
                    "embed_model": embed_key,
                    "site_key": site_key,
                    "num_chunks": timings["num_chunks"],
                    "extract_sec": f"{timings['extract_sec']:.2f}",
                    "embed_sec": f"{timings['embed_sec']:.2f}",
                    "llm_model": args.llm_model,
                    "llm_load_sec": f"{llm_load_time:.2f}",
                })

                if index is None:
                    print(f"      No index for {site_key}, skipping.")
                    continue

                for q_id, question in enumerate(questions):
                    print(f"      Q{q_id}: {question[:50]}...")
                    chunks = retrieve(index, question)
                    answer, gen_time = generate_answer(model, tokenizer, question, chunks, site_key, args.llm_model)
                    verification = verify_answer(answer, chunks)

                    results_rows.append({
                        "extractor": extractor_name,
                        "embed_model": embed_key,
                        "llm_model": args.llm_model,
                        "site_key": site_key,
                        "question_id": q_id,
                        "question": question,
                        "answer": answer.text,
                        "citation_verified": verification.is_valid,
                        "gen_time_sec": f"{gen_time:.2f}",
                    })
                    print(f"        A ({gen_time:.1f}s): {answer.text[:80]}...")

    # Write results
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    with open(timing_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"\nResults: {results_file}")
    print(f"Timings: {timing_file}")


if __name__ == "__main__":
    main()

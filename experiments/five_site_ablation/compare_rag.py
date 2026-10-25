"""
RAG on original PDFs vs condensed text — one site per run, both sources on the same GPU.

Same pipeline and metrics as benchmark_rag.py (Experiment 2, extraction comparison), so the
numbers line up with the August benchmarks:
  - per-site timing: num_docs, num_chunks, extract_sec, embed_sec, embed_load_sec, llm_load_sec
  - per-question:    answer, citation_verified, gen_time_sec, retrieval_ms, answer length,
                     NOT PRESENT, repetition, retrieved sources

Sources:
  pdf        original PDFs (five_sites/<folder>), text extracted with pypdf at index time,
             exactly as the current RAG does
  condensed  the condenser's cleaned text (<condensed_root>/<EPA ID>/<condenser>/clean/text/*.txt):
             duplicates dropped (Jaccard >= 0.98, lengths within 5%), headers/footers kept since the
             10-16-26 re-clean (stripped before), tables kept; original page numbers kept

Both sources run in one job with the models loaded once. Retrieval can be dense, BM25, or
hybrid dense+BM25 with deterministic reciprocal-rank fusion. Qwen3 dense retrieval can use
the versioned Superfund query instruction while documents remain unmodified. Extraction,
chunks, dense indexes, and BM25 indexes use a shared content-addressed cache; generated
answers remain isolated in each arm's output directory. Citation evaluation is explicitly
versioned as legacy, lenient, or strict and never claims to check semantic entailment.
"""
import argparse
import json
import os
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def find_project_root():
    override = os.environ.get("SUPERFUND_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "benchmark_rag.py").is_file() or not (root / "rag_march6").is_dir():
            raise RuntimeError(
                "SUPERFUND_PROJECT_ROOT must contain benchmark_rag.py and rag_march6/"
            )
        return str(root)
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "benchmark_rag.py").is_file() and (candidate / "rag_march6").is_dir():
            return str(candidate)
    raise RuntimeError(
        "Could not locate benchmark_rag.py and rag_march6/. "
        "Set SUPERFUND_PROJECT_ROOT explicitly."
    )


BASE_DIR = find_project_root()
sys.path.insert(0, BASE_DIR)

import torch  # noqa: E402
import faiss as faiss_native  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from benchmark_rag import EMBEDDING_MODELS, LLM_MODELS, CHUNK_SIZE, CHUNK_OVERLAP, MAX_PAGES, TOP_K  # noqa: E402
from benchmark_rag import load_llm, generate_answer  # noqa: E402
from rag_march6.retrieve import RetrievedChunk, retrieve  # noqa: E402
from rag_march6.verify import verify_answer as legacy_verify_answer  # noqa: E402
from bm25_rrf import TOKENIZER_POLICIES  # noqa: E402
from citation_eval import verify_text as contract_verify_text  # noqa: E402
from embedding_modes import (  # noqa: E402
    QUERY_MODES,
    QUERY_MODE_NONE,
    QueryAwareEmbeddings,
    build_embedding_profile,
    manifest_dict as embedding_manifest_dict,
)
from retrieval_modes import (  # noqa: E402
    RETRIEVAL_MODES,
    RetrievalSettings,
    get_or_build_bm25,
    retrieve_question,
)
from question_io import load_questions  # noqa: E402
from rag_cache import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    atomic_write_csv,
    config_key,
    fingerprint_source,
    get_or_build_directory,
    get_or_build_json_gzip,
    sha256_file,
)

PAGE_MARK = re.compile(r"^<!-- Page (\d+) -->\s*$", re.M)
EXTRACTION_SCHEMA = "superfund-extraction-v1"
CHUNK_SCHEMA = "superfund-char-window-v1"
INDEX_SCHEMA = "superfund-faiss-v1"


def event_field(event, name, default=""):
    return getattr(event, name, default) if event is not None else default


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


# ── Text sources: both return [(doc_id, [(page_no, text), ...]), ...] ─────────
def pages_from_pdfs(pdf_dir, max_pages):
    from pypdf import PdfReader
    docs = []
    for f in sorted(os.listdir(pdf_dir)):
        if not f.lower().endswith(".pdf"):
            continue
        pages = []
        try:
            reader = PdfReader(os.path.join(pdf_dir, f))
            for i, page in enumerate(reader.pages):
                if max_pages and i >= max_pages:
                    break
                try:
                    pages.append((i + 1, (page.extract_text() or "").strip()))
                except Exception:
                    pages.append((i + 1, ""))
        except Exception as e:
            print(f"    pypdf failed on {f}: {e}")
        docs.append((os.path.splitext(f)[0], pages))
    return docs


def pages_from_condensed(text_dir, max_pages):
    docs = []
    for f in sorted(os.listdir(text_dir)):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(text_dir, f), encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        parts = PAGE_MARK.split(body)  # [header, page_no, text, page_no, text, ...]
        pages = [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
        if max_pages:
            pages = [p for p in pages if p[0] <= max_pages]
        docs.append((os.path.splitext(f)[0], pages))
    return docs


# ── Chunking: same windows as benchmark_rag.chunk_text, plus the page each chunk starts on ──
def chunk_doc(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text, starts = "", []
    for no, t in pages:
        if t:
            starts.append((len(text), no))
            text += t + "\n\n"
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size]
        page = next((no for off, no in reversed(starts) if off <= i), starts[0][1] if starts else 0)
        out.append((piece, page))
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def build_documents(site, docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    out = []
    for doc_id, pages in docs:
        for k, (chunk, page) in enumerate(chunk_doc(pages, size=size, overlap=overlap)):
            if chunk.strip():
                out.append(Document(page_content=chunk, metadata={
                    "site_key": site, "pdf_filename": doc_id + ".pdf", "page": page, "chunk_index": k}))
    return out


def docs_to_payload(docs):
    return [{"doc_id": doc_id, "pages": [[page, text] for page, text in pages]}
            for doc_id, pages in docs]


def docs_from_payload(payload):
    return [(item["doc_id"], [(int(page), text) for page, text in item["pages"]])
            for item in payload]


def chunks_to_payload(chunks):
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunks]


def chunks_from_payload(payload):
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in payload]


# ── Answer metrics (as in the Experiment 1 quality table) ────────────────────
def clean_answer(text):
    return re.sub(r"^\s*thought\s*\n", "", text).strip()  # gemma4 leaves a "thought" header


def is_repetitive(text, n=5, times=3):
    words = text.lower().split()
    grams = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= times:
            return True
    return False


def run_source(
    args,
    source,
    embeddings,
    embedding_profile,
    model,
    tokenizer,
    llm_load_sec,
    embed_load_sec,
    questions,
    settings,
):
    tag = f"{args.site}_{source}"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    variant = args.condenser if source == "condensed" else "pypdf"
    source_dir = (args.pdf_dir if source == "pdf" else
                  os.path.join(args.condensed_root, args.site, args.condenser, "clean", "text"))
    suffix = ".pdf" if source == "pdf" else ".txt"

    # 1. Fingerprint source bytes. Full file hashes are memoized behind a
    # path/size/mtime inventory; --force_content_rehash bypasses that memo.
    content_sha256, fingerprint_info = fingerprint_source(
        source_dir, suffix, args.cache_dir, force_rehash=args.force_content_rehash)
    extraction_config = {
        "schema": EXTRACTION_SCHEMA,
        "cache_format": CACHE_FORMAT_VERSION,
        "site": args.site,
        "source": source,
        "variant": variant,
        "content_sha256": content_sha256,
        "max_pages": args.max_pages or None,
        "parser_version": package_version("pypdf") if source == "pdf" else "page-mark-v1",
    }

    def build_extraction_payload():
        if source == "pdf":
            built_docs = pages_from_pdfs(args.pdf_dir, args.max_pages)
        else:
            built_docs = pages_from_condensed(source_dir, args.max_pages)
        return docs_to_payload(built_docs)

    docs_payload, extraction_event = get_or_build_json_gzip(
        args.cache_dir, "extraction", extraction_config, build_extraction_payload)
    docs = docs_from_payload(docs_payload)
    n_pages = sum(len(pages) for _, pages in docs)
    n_chars = sum(len(text) for _, pages in docs for _, text in pages)

    # 2. Chunk cache depends on the exact extraction artifact plus the window
    # settings. A future chunking ablation therefore reuses extraction safely.
    chunk_config = {
        "schema": CHUNK_SCHEMA,
        "cache_format": CACHE_FORMAT_VERSION,
        "site": args.site,
        "source": source,
        "extraction_key": extraction_event.key,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }

    def build_chunk_payload():
        return chunks_to_payload(build_documents(
            args.site,
            docs,
            size=args.chunk_size,
            overlap=args.chunk_overlap,
        ))

    chunks_payload, chunk_event = get_or_build_json_gzip(
        args.cache_dir, "chunks", chunk_config, build_chunk_payload)
    chunks = chunks_from_payload(chunks_payload)
    if not chunks:
        raise RuntimeError(f"{tag}: no non-empty chunks were produced")
    print(
        f"{tag}: {len(docs)} docs, {n_pages} pages, {n_chars:,} chars, {len(chunks)} chunks; "
        f"fingerprint={'hit' if fingerprint_info['hit'] else 'built'} "
        f"extract={'hit' if extraction_event.hit else 'built'} "
        f"chunks={'hit' if chunk_event.hit else 'built'}",
        flush=True,
    )

    # 3. Build only the indexes required by this retrieval arm. The dense cache
    # key includes the exact query profile because FAISS stores an embedding
    # object that controls query-time transformation after reload.
    dense_index = None
    dense_event = None
    if settings.needs_dense:
        index_config = {
            "schema": INDEX_SCHEMA,
            "cache_format": CACHE_FORMAT_VERSION,
            "site": args.site,
            "source": source,
            "chunk_key": chunk_event.key,
            "embed_model": args.embed_model,
            "embed_model_path": os.path.realpath(EMBEDDING_MODELS[args.embed_model]),
            **embedding_manifest_dict(embedding_profile),
            "normalize_embeddings": True,
            "encode_batch_size": 8,
            "langchain_community_version": package_version("langchain-community"),
            "langchain_huggingface_version": package_version("langchain-huggingface"),
            "sentence_transformers_version": package_version("sentence-transformers"),
            "transformers_version": package_version("transformers"),
            "torch_version": torch.__version__,
            "faiss_version": getattr(faiss_native, "__version__", "unknown"),
        }

        def load_index(path):
            return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)

        def build_index(path):
            built_index = FAISS.from_documents(chunks, embeddings)
            built_index.save_local(str(path))
            return built_index, {"num_vectors": int(built_index.index.ntotal)}

        dense_index, dense_event = get_or_build_directory(
            args.cache_dir, "faiss", index_config, load_index, build_index)
        print(
            f"  FAISS {'loaded' if dense_event.hit else 'built'}: {dense_index.index.ntotal} vectors "
            f"in {dense_event.elapsed_sec:.1f}s ({dense_event.path})",
            flush=True,
        )

    bm25_index = None
    bm25_event = None
    if settings.needs_bm25:
        bm25_index, bm25_event = get_or_build_bm25(
            args.cache_dir,
            chunk_event.key,
            chunks,
            settings.bm25_config,
        )
        details = bm25_event.metadata.get("details", {})
        print(
            f"  BM25 {'loaded' if bm25_event.hit else 'built'}: "
            f"{bm25_index.document_count} documents, {len(bm25_index.postings)} terms, "
            f"{details.get('posting_pairs', 'unknown')} postings in {bm25_event.elapsed_sec:.1f}s "
            f"({bm25_event.path})",
            flush=True,
        )

    questions_sha256 = sha256_file(args.questions_file)
    run_fingerprint = config_key({
        "site": args.site,
        "source": source,
        "extraction_key": extraction_event.key,
        "chunk_key": chunk_event.key,
        "dense_index_key": event_field(dense_event, "key"),
        "bm25_index_key": event_field(bm25_event, "key"),
        "embedding_profile": embedding_manifest_dict(embedding_profile),
        "retrieval": settings.as_dict(),
        "verifier_mode": args.verifier_mode,
        "questions_sha256": questions_sha256,
        "llm_model": args.llm_model,
        "prompt": args.prompt,
        "top_k": settings.top_k,
    })

    # 4. Questions. Answers are never cached across arms; only deterministic
    # corpus/index work is shared. The run fingerprint makes stale output clear.
    rows = []
    for question_order, (question_id, question) in enumerate(questions):
        question_started = time.time()
        t0 = time.time()
        got = retrieve_question(
            settings,
            question=question,
            chunks=chunks,
            retrieved_chunk_type=RetrievedChunk,
            dense_index=dense_index,
            bm25_index=bm25_index,
            dense_retrieve=retrieve,
        )
        retrieval_ms = (time.time() - t0) * 1000
        answer, gen_time = generate_answer(model, tokenizer, question, got, args.site_name, args.llm_model)
        text = clean_answer(answer.text)
        verify_started = time.time()
        if args.verifier_mode == "legacy":
            ver = legacy_verify_answer(answer, got)
            citation_status = "legacy_valid" if ver.is_valid else "legacy_invalid"
            citation_verified = ver.is_valid
            citation_count = len(ver.verified_sources) + len(ver.failed_citations)
            parse_issues = []
            reason_codes = []
            is_abstention = text == "NOT PRESENT"
            semantic_support_checked = False
        else:
            ver = contract_verify_text(text, got, mode=args.verifier_mode)
            citation_status = ver.status
            citation_verified = ver.citation_verified
            citation_count = ver.citation_count
            parse_issues = ver.parse_issues
            reason_codes = ver.reason_codes
            is_abstention = ver.is_abstention
            semantic_support_checked = ver.semantic_support_checked
        verification_ms = (time.time() - verify_started) * 1000
        if args.verifier_mode == "strict" and not ver.is_valid:
            answer_disposition = "rejected"
            released_answer = "INSUFFICIENT_EVIDENCE"
        elif is_abstention:
            answer_disposition = "abstained"
            released_answer = text
        elif ver.is_valid:
            answer_disposition = "accepted"
            released_answer = text
        else:
            # Legacy/lenient arms retain their historical visible answer while
            # making the failed contract explicit for analysis.
            answer_disposition = "flagged"
            released_answer = text
        rows.append({
            "source": source, "condenser": args.condenser if source == "condensed" else "pypdf",
            "embed_model": args.embed_model, "llm_model": args.llm_model, "site_key": args.site,
            "embedding_model_path": (
                os.path.realpath(EMBEDDING_MODELS[args.embed_model]) if settings.needs_dense else ""
            ),
            "site_name": args.site_name, "prompt": args.prompt,
            "question_id": question_id, "question_order": question_order, "question": question,
            "raw_response": getattr(answer, "raw_response", ""), "answer": text,
            "released_answer": released_answer,
            "answer_disposition": answer_disposition,
            "run_fingerprint": run_fingerprint,
            "extraction_cache_key": extraction_event.key, "chunk_cache_key": chunk_event.key,
            "dense_index_cache_key": event_field(dense_event, "key"),
            "bm25_index_cache_key": event_field(bm25_event, "key"),
            "query_instruction_mode": embedding_profile.query_mode,
            "query_instruction": embedding_profile.instruction or "",
            "query_instruction_sha256": embedding_profile.instruction_sha256,
            "embedding_index_key": embedding_profile.index_key,
            "retrieval_mode": settings.mode,
            "retrieval_candidate_k": settings.candidate_k,
            "final_top_k": settings.top_k,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "bm25_tokenizer": settings.bm25_tokenizer,
            "fusion": settings.as_dict()["fusion"],
            "rrf_k": settings.rrf_k,
            "retrieval_score_semantics": settings.score_semantics,
            # citation_verified is deliberately tri-state: blank/None for an
            # exact NOT PRESENT abstention under the new verifier.
            "verifier_mode": args.verifier_mode,
            "citation_status": citation_status,
            "citation_verified": citation_verified,
            "num_citations": citation_count,
            "failed_citations": "; ".join(ver.failed_citations),
            "citation_parse_issues": "; ".join(parse_issues),
            "citation_reason_codes": "; ".join(reason_codes),
            "semantic_support_checked": semantic_support_checked,
            "not_present": is_abstention,
            "answer_chars": len(text), "repetitive": is_repetitive(text),
            "retrieval_ms": f"{retrieval_ms:.1f}", "gen_time_sec": f"{gen_time:.2f}",
            "verification_ms": f"{verification_ms:.3f}",
            "question_total_sec": f"{time.time() - question_started:.3f}",
            "retrieved_sources": " | ".join(f"{c.pdf_filename}:p{c.page}" for c in got),
            "retrieved_scores": " | ".join(f"{float(c.score):.3f}" for c in got),
            "evidence_quote": answer.evidence_quote,
        })
        print(
            f"  {question_id} ({gen_time:.1f}s, retrieval {retrieval_ms:.0f} ms): {text[:100]!r}",
            flush=True,
        )

    timing = {
        "source": source, "condenser": rows[0]["condenser"], "embed_model": args.embed_model,
        "embedding_model_path": (
            os.path.realpath(EMBEDDING_MODELS[args.embed_model]) if settings.needs_dense else ""
        ),
        "site_key": args.site, "num_docs": len(docs), "num_pages": n_pages, "num_chars": n_chars,
        "num_chunks": len(chunks),
        "content_sha256": content_sha256,
        "source_fingerprint_cache_hit": fingerprint_info["hit"],
        "source_fingerprint_sec": f"{fingerprint_info['elapsed_sec']:.2f}",
        "extraction_cache_hit": extraction_event.hit, "extraction_cache_key": extraction_event.key,
        "extract_sec": f"{extraction_event.elapsed_sec:.2f}",
        "extract_build_sec": f"{extraction_event.build_sec:.2f}",
        "chunk_cache_hit": chunk_event.hit, "chunk_cache_key": chunk_event.key,
        "chunk_sec": f"{chunk_event.elapsed_sec:.2f}", "chunk_build_sec": f"{chunk_event.build_sec:.2f}",
        "dense_index_cache_hit": event_field(dense_event, "hit"),
        "dense_index_cache_key": event_field(dense_event, "key"),
        "dense_index_sec": f"{event_field(dense_event, 'elapsed_sec', 0.0):.2f}",
        "dense_index_build_sec": f"{event_field(dense_event, 'build_sec', 0.0):.2f}",
        "bm25_index_cache_hit": event_field(bm25_event, "hit"),
        "bm25_index_cache_key": event_field(bm25_event, "key"),
        "bm25_index_sec": f"{event_field(bm25_event, 'elapsed_sec', 0.0):.2f}",
        "bm25_index_build_sec": f"{event_field(bm25_event, 'build_sec', 0.0):.2f}",
        "run_fingerprint": run_fingerprint,
        "embed_load_sec": f"{embed_load_sec:.2f}", "llm_model": args.llm_model, "llm_load_sec": f"{llm_load_sec:.2f}",
        "max_pages": args.max_pages, "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
        **embedding_manifest_dict(embedding_profile),
        "retrieval_mode": settings.mode,
        "retrieval_candidate_k": settings.candidate_k,
        "final_top_k": settings.top_k,
        "bm25_k1": settings.bm25_k1,
        "bm25_b": settings.bm25_b,
        "bm25_tokenizer": settings.bm25_tokenizer,
        "fusion": settings.as_dict()["fusion"],
        "rrf_k": settings.rrf_k,
        "retrieval_score_semantics": settings.score_semantics,
        "verifier_mode": args.verifier_mode,
        "peak_gpu_allocated_gb": (
            f"{torch.cuda.max_memory_allocated() / 1e9:.3f}" if torch.cuda.is_available() else "0.000"
        ),
        "peak_gpu_reserved_gb": (
            f"{torch.cuda.max_memory_reserved() / 1e9:.3f}" if torch.cuda.is_available() else "0.000"
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }
    for name, data in ((f"answers_{tag}.csv", rows), (f"timing_{tag}.csv", [timing])):
        atomic_write_csv(os.path.join(args.output_dir, name), list(data[0]), data)
    print(json.dumps(timing), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="EPA ID")
    ap.add_argument("--site_name", required=True)
    ap.add_argument("--sources", nargs="+", default=["pdf", "condensed"], choices=["pdf", "condensed"])
    ap.add_argument("--pdf_dir", required=True, help="original PDFs (five_sites/<folder>)")
    ap.add_argument("--condensed_root", required=True, help=".../condensing mining 10-15-26/multisite")
    ap.add_argument("--condenser", default="pymupdf_smart")
    ap.add_argument("--questions_file", required=True)
    ap.add_argument("--embed_model", default="e5-mistral-7b", choices=["none", *EMBEDDING_MODELS])
    ap.add_argument(
        "--query_instruction_mode",
        choices=QUERY_MODES,
        default=QUERY_MODE_NONE,
        help="none preserves raw queries; superfund applies the versioned Qwen3 query instruction",
    )
    ap.add_argument(
        "--query_instruction",
        help="optional custom Qwen3 task description; requires --query_instruction_mode superfund",
    )
    ap.add_argument("--llm_model", default="gemma4-12b", choices=list(LLM_MODELS))
    ap.add_argument("--max_pages", type=int, default=0, help="pages per document; 0 = all (default)")
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--retrieval_mode", choices=RETRIEVAL_MODES, default="dense")
    ap.add_argument("--candidate_k", type=int, default=30)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--bm25_k1", type=float, default=1.2)
    ap.add_argument("--bm25_b", type=float, default=0.75)
    ap.add_argument("--bm25_tokenizer", choices=TOKENIZER_POLICIES, default=TOKENIZER_POLICIES[0])
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--verifier_mode", choices=["legacy", "lenient", "strict"], default="lenient")
    ap.add_argument("--cache_dir", help="shared content-addressed extraction/chunk/FAISS cache")
    ap.add_argument("--index_dir", help="deprecated alias for --cache_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embedding_profile", default=None,
                    help="deprecated; profiles are now derived from model and query-instruction settings")
    ap.add_argument("--force_content_rehash", action="store_true",
                    help="rehash every source byte instead of trusting the stat-keyed hash memo")
    ap.add_argument("--prompt", default="current", choices=["current", "partial"],
                    help="current = rag_march6.build_prompt (strict NOT PRESENT); partial = probe_answers.partial_prompt")
    ap.add_argument("--reuse_index", action="store_true",
                    help="deprecated no-op; compatible artifacts are always reused from the content-addressed cache")
    args = ap.parse_args()
    args.cache_dir = args.cache_dir or args.index_dir
    if not args.cache_dir:
        ap.error("one of --cache_dir or --index_dir is required")
    if args.chunk_size <= 0:
        ap.error("--chunk_size must be positive")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        ap.error("--chunk_overlap must be >= 0 and smaller than --chunk_size")
    if args.retrieval_mode == "bm25" and args.embed_model != "none":
        ap.error("BM25-only requires --embed_model none so no dense model is loaded")
    if args.retrieval_mode in {"dense", "hybrid"} and args.embed_model == "none":
        ap.error("dense and hybrid retrieval require an embedding model")
    if args.retrieval_mode == "bm25" and args.query_instruction_mode != QUERY_MODE_NONE:
        ap.error("BM25-only does not accept a dense query instruction")
    if args.prompt == "partial":
        # generate_answer calls benchmark_rag.build_prompt; swap in the partial-answer prompt from the probe
        import benchmark_rag
        try:
            from rag_condensed_compare.probe_answers import partial_prompt
        except ModuleNotFoundError:
            from probe_answers import partial_prompt
        benchmark_rag.build_prompt = partial_prompt

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        questions = load_questions(args.questions_file)
    except ValueError as exc:
        ap.error(str(exc))
    settings = RetrievalSettings(
        mode=args.retrieval_mode,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_tokenizer=args.bm25_tokenizer,
        rrf_k=args.rrf_k,
    )
    try:
        settings.as_dict()
        embedding_profile = build_embedding_profile(
            args.embed_model,
            args.query_instruction_mode,
            args.query_instruction,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.embedding_profile and args.embedding_profile != embedding_profile.index_key:
        ap.error(
            "--embedding_profile is deprecated and must equal the derived embedding_index_key "
            f"({embedding_profile.index_key}) when supplied"
        )
    print(
        f"{args.site}: sources={args.sources}, questions={len(questions)}, "
        f"retrieval={settings.mode}, embed={args.embed_model}, "
        f"query_instruction_mode={embedding_profile.query_mode}, "
        f"embedding_index_key={embedding_profile.index_key}, verifier={args.verifier_mode}, "
        f"llm={args.llm_model}, max_pages={args.max_pages}, "
        f"candidate_k={settings.candidate_k}, top_k={settings.top_k}",
        flush=True,
    )

    embeddings = None
    embed_load_sec = 0.0
    if settings.needs_dense:
        t0 = time.time()
        base_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODELS[args.embed_model],
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"batch_size": 8, "normalize_embeddings": True},
        )
        embeddings = (
            base_embeddings
            if embedding_profile.query_mode == QUERY_MODE_NONE
            else QueryAwareEmbeddings(base_embeddings, embedding_profile)
        )
        embed_load_sec = time.time() - t0
        embeddings.embed_documents(["warm-up " * 300] * 8)  # CUDA warm-up, not timed
        embeddings.embed_query("warm-up")
    model, tokenizer, llm_load_sec = load_llm(args.llm_model)
    print(f"  embeddings loaded in {embed_load_sec:.1f}s, LLM in {llm_load_sec:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    for source in args.sources:
        run_source(
            args,
            source,
            embeddings,
            embedding_profile,
            model,
            tokenizer,
            llm_load_sec,
            embed_load_sec,
            questions,
            settings,
        )
    print("done")


if __name__ == "__main__":
    main()

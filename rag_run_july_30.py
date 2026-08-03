"""
Superfund RAG — batch runner
Loops over LLM models and a PDF directory, writes answers to a CSV.

# ── Quick start ───────────────────────────────────────────────────────────────
# Activate the venv first:
#   source .venv/bin/activate
#
# Run with defaults (all models, built-in questions, full pdfs/ folder):
#   python rag_run.py
#
# Use a custom questions file (one question per line, # = comment):
#   python rag_run.py --questions_file questions.txt
#
# Run only specific models:
#   python rag_run.py --llm_models qwen3-8b llama3-8b
#
# Point at a different PDF folder (e.g. last-10 subset):
#   python rag_run.py --pdfs_dir /path/to/subset_pdfs
#
# Filter PDFs by filename prefix instead of copying files:
#   python rag_run.py --pdfs_filter Wisconsin_Tomah Wyoming
#   (keeps only PDFs whose filename starts with one of the given prefixes)
# 
# Run with Docling: 
#   python rag_run.py --use_docling
#
# Full example — last-10 papers, two models, custom questions:
#   python rag_run.py \
#       --pdfs_filter Wisconsin_Tomah Wisconsin_Unity Wisconsin_Waste \
#                     Wisconsin_Wausau Wyoming \
#       --llm_models qwen3-8b llama3-8b \
#       --questions_file questions.txt
# ─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

import torch
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer, pipeline

# ── Model registry ────────────────────────────────────────────────────────────
HF_HUB = "/datasets/ai"

EMBEDDING_MODELS = {
    # Qwen3 causal-LM embeddings (last-token pooling) — use --embed_model qwen3-embed-8b or qwen3-embed-0.6b
    "qwen3-embed-8b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    # lightweight option
    "qwen3-embed-0.6b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-0.6B/snapshots",
    # sentence-transformers compatible
    "e5-mistral-7b": f"{HF_HUB}/intfloat/hub/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1",
}

LLM_MODELS = {
    # Qwen3 base (fits on V100 16GB)
    "qwen3-0.6b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-0.6B/snapshots",
    "qwen3-1.7b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-1.7B/snapshots",
    "qwen3-4b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-4B/snapshots",
    "qwen3-8b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-8B/snapshots",
    # Qwen3 thinking/instruct variants
    "qwen3-4b-thinking": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots",
    # MoE: activates ~3B, fits ~10GB
    "qwen3-30b-thinking": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots",
    # MoE: activates ~3B, fits ~8GB
    "qwen3-30b-moe": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-30B-A3B/snapshots",
    # Larger (needs A100/L40S 40GB+)
    "qwen3-14b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-14B/snapshots",
    "qwen3-32b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-32B/snapshots",
    # Qwen2.5 for comparison
    "qwen2.5-7b": f"{HF_HUB}/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots",
    "qwen2.5-14b": f"{HF_HUB}/qwen2/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots",
    # Llama3 for cross-family comparison
    "llama3-8b": f"{HF_HUB}/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots",
    # Gemma-4
    "gemma4-31B": f"{HF_HUB}/gemma/hub/models--google--gemma-4-31B/snapshots",
    "gemma4-12B-it": f"{HF_HUB}/gemma/hub/models--google--gemma-4-12B-it/snapshots",
    "gemma4-26b-a4b": f"{HF_HUB}/gemma/hub/models--google--gemma-4-26b-a4b/snapshots",
}

DEFAULT_QUESTIONS = [
    "What contaminants are present at this site?",
    "What is the location of this Superfund site?",
    "What are the health risks associated with this site?",
    "What cleanup actions have been taken?",
]

# ── Config ────────────────────────────────────────────────────────────────────
MAX_PAGES_PER_PDF = 50  # was 25 — extract more pages per PDF
MAX_CHARS_PER_PDF = 80_000  # was 40K — capture more site content
CHUNK_SIZE_CHARS = 2400  # was 1800 — larger chunks = less fragmentation
CHUNK_OVERLAP_CHARS = 300  # was 250
KEYWORD_SNIPPET_CHARS = 260


# ── Qwen3 embeddings (last-token pooling over causal LM) ─────────────────────
class Qwen3Embeddings(Embeddings):
    """Wraps any Qwen3 causal LM for use as an embedding model via last-token pooling."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        from transformers import PreTrainedTokenizerBase

        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.model = (
            AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        sanitized = [t.strip() if (isinstance(t, str) and t.strip()) else "document" for t in texts]
        encoded = self.tokenizer(
            sanitized, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        if encoded["input_ids"].ndim < 2 or encoded["input_ids"].shape[1] == 0:
            eos_id = self.tokenizer.eos_token_id or 0
            encoded = {
                "input_ids": torch.full((len(texts), 1), eos_id, dtype=torch.long, device=self.device),
                "attention_mask": torch.ones((len(texts), 1), dtype=torch.long, device=self.device),
            }
        with torch.no_grad():
            out = self.model(**encoded)
        # last non-padding token
        seq_len = torch.clamp(encoded["attention_mask"].sum(dim=1) - 1, min=0)
        vecs = out.last_hidden_state[torch.arange(len(texts)), seq_len]
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
        return vecs.cpu().float().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # batch to avoid OOM
        results = []
        for i in range(0, len(texts), 16):
            results.extend(self._embed(texts[i : i + 16]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class DoclingDocumentParser:
    def __init__(self, model_path: str, max_tokens: int = 512):
        # Docling Config
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.do_picture_description = False

        self.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
        self.pipeline_options.accelerator_options = self.accelerator_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer, max_tokens=max_tokens, merge_peers=True
        )

    def extract_and_chunk(self, path: str) -> list[str]:
        try:
            doc = self.converter.convert(path).document
            chunks = list(self.chunker.chunk(doc))
            return [chunk.text for chunk in chunks]
        except Exception as e:
            print(f"Error parsing {path} with docling: {e}")
            return []


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8).startswith(b"%PDF-")
    except Exception:
        return False


def _iter_pdfs(pdf_dir: str) -> Iterable[str]:
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith(".pdf"):
            yield os.path.join(pdf_dir, fn)


def _site_key(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return re.split(r"_additional_\d+$", stem)[0]


def _resolve_snapshot(model_path: str) -> str:
    """If path ends in /snapshots, pick the first (only) snapshot hash."""
    if model_path.endswith("/snapshots"):
        snaps = os.listdir(model_path)
        if snaps:
            return os.path.join(model_path, snaps[0])
    return model_path


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


# ── PDF extraction ────────────────────────────────────────────────────────────


def extract_pdf_text(path: str, max_pages: int = MAX_PAGES_PER_PDF) -> tuple[bool, str]:
    """Returns (ok, text)."""
    if not _looks_like_pdf(path):
        return False, ""
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return False, ""
        parts = []
        for page in reader.pages[:max_pages]:
            try:
                t = (page.extract_text() or "").strip()
            except Exception:
                t = ""
            if t:
                parts.append(t)
        return True, "\n\n".join(parts).strip()
    except Exception:
        return False, ""


# ── Index builder ─────────────────────────────────────────────────────────────
def build_index(
    pdfs_dir: str, embed_model_path: str, pdf_prefixes: list[str] | None = None
):
    """Extract PDFs, chunk, embed, return (vectorstore, site_fulltext).
    pdf_prefixes: if set, only index PDFs whose filename starts with one of these strings.
    """
    site_texts: dict[str, list[str]] = defaultdict(list)
    for path in _iter_pdfs(pdfs_dir):
        if pdf_prefixes and not any(
            os.path.basename(path).startswith(p) for p in pdf_prefixes
        ):
            continue
        ok, text = extract_pdf_text(path)
        if ok and text:
            site_texts[_site_key(path)].append(text)

    docs: list[Document] = []
    site_fulltext: dict[str, str] = {}

    for sk, parts in site_texts.items():
        merged = "\n\n".join(parts)[:MAX_CHARS_PER_PDF]
        site_fulltext[sk] = merged
        for idx, chunk in enumerate(
            _chunk(merged, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        ):
            cleaned = _clean(chunk)
            if not cleaned:
                continue
            docs.append(
                Document(
                    page_content=f"SITE KEY: {sk}\n\n{cleaned}",
                    metadata={"site_key": sk, "chunk_index": idx},
                )
            )

    print(f"  Indexed {len(site_texts)} sites, {len(docs)} chunks")
    # Use Qwen3Embeddings for qwen3-embed-* models, HuggingFaceEmbeddings for others
    if "qwen3-embed" in embed_model_path or "Qwen3" in embed_model_path:
        embeddings = Qwen3Embeddings(embed_model_path)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, site_fulltext


def build_index_docling(
    pdfs_dir: str, embed_model_path: str, pdf_prefixes: list[str] | None = None
):
    """Extract PDFs, chunk, embed using Docling, return (vectorstore, site_fulltext)."""
    parser = DoclingDocumentParser(model_path=embed_model_path)

    site_chunks: dict[str, list[str]] = defaultdict(list)
    site_fulltext: dict[str, str] = {}

    for path in _iter_pdfs(pdfs_dir):
        if pdf_prefixes and not any(
            os.path.basename(path).startswith(p) for p in pdf_prefixes
        ):
            continue
        sk = _site_key(path)
        chunks_text = parser.extract_and_chunk(path)
        if chunks_text:
            site_chunks[sk].extend(chunks_text)

    docs: list[Document] = []

    for sk, chunks in site_chunks.items():
        merged = "\n\n".join(chunks)[:MAX_CHARS_PER_PDF]
        site_fulltext[sk] = merged
        for idx, chunk_text in enumerate(chunks):
            cleaned = _clean(chunk_text)
            if not cleaned:
                continue
            docs.append(
                Document(
                    page_content=f"SITE KEY: {sk}\n\n{cleaned}",
                    metadata={"site_key": sk, "chunk_index": idx},
                )
            )

    print(f"  Indexed {len(site_chunks)} sites, {len(docs)} chunks (using Docling)")
    # Use Qwen3Embeddings for qwen3-embed-* models, HuggingFaceEmbeddings for others
    if "qwen3-embed" in embed_model_path or "Qwen3" in embed_model_path:
        embeddings = Qwen3Embeddings(embed_model_path)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, site_fulltext


# ── RAG query ─────────────────────────────────────────────────────────────────
def ask(
    question: str,
    site_key: str,
    site_fulltext: dict[str, str],
    vectorstore,
    llm,
    tokenizer=None,
    top_k: int = 6,
) -> str:
    """Answer a question using context from a single site only."""
    # Retrieve chunks filtered to this site
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 3})
    retrieved = [
        d for d in retriever.invoke(question) if d.metadata.get("site_key") == site_key
    ][:top_k]

    # If fewer than 3 chunks retrieved, widen the search
    if len(retrieved) < 3:
        retriever2 = vectorstore.as_retriever(search_kwargs={"k": top_k * 6})
        retrieved = [
            d
            for d in retriever2.invoke(question)
            if d.metadata.get("site_key") == site_key
        ][:top_k]

    # Ctrl+F fallback if semantic retrieval missed this site
    if not retrieved:
        text = site_fulltext.get(site_key, "")
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        for term in q_tokens:
            m = re.search(re.escape(term), text, re.IGNORECASE)
            if m:
                s = max(0, m.start() - KEYWORD_SNIPPET_CHARS)
                e = min(len(text), m.end() + KEYWORD_SNIPPET_CHARS)
                retrieved = [
                    Document(
                        page_content=f"SITE KEY: {site_key}\n\n{text[s:e]}",
                        metadata={"site_key": site_key},
                    )
                ]
                break

    if not retrieved:
        return "No relevant information found in the PDFs for this site."

    context = "\n\n".join(d.page_content for d in retrieved)
    system_content = (
        f"You are a document extraction system. Use ONLY the provided file: {site_key}. "
        f"If the answer is not explicitly stated in the file, output exactly: NOT PRESENT. "
        f"Output a single short answer only. Do NOT include explanations, reasoning, or extra text."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{system_content}\n\n{user_content}\n\nAnswer:"
    else:
        prompt = f"{system_content}\n\n{user_content}\n\nAnswer:"

    raw = llm.invoke(prompt)

    # Strip thinking/reasoning blocks (e.g., <think>...</think>)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Strip prompt echo prefixes if present
    if "Answer:" in raw:
        raw = raw.split("Answer:")[-1]
    if "Assistant:" in raw:
        raw = raw.split("Assistant:")[-1]

    return raw.strip()



# ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdfs_dir",
        default="/work/pi_larsonj_wit_edu/superfund/project/superfund/pdfs/pdfs",
    )
    parser.add_argument(
        "--out_dir",
        default="/work/pi_larsonj_wit_edu/superfund/project/superfund/pdf_only_output",
    )
    parser.add_argument(
        "--questions_file",
        default="questions.txt",
        help="Path to questions file (one per line, # = comment). Default: questions.txt",
    )
    parser.add_argument(
        "--llm_models",
        nargs="+",
        default=list(LLM_MODELS.keys()),
        help=f"Model keys to run: {list(LLM_MODELS.keys())}",
    )
    parser.add_argument(
        "--embed_model",
        default="e5-mistral-7b",
        help=f"Embedding model key: {list(EMBEDDING_MODELS.keys())}",
    )
    parser.add_argument(
        "--pdfs_filter",
        nargs="+",
        default=None,
        help="Only index PDFs whose filename starts with one of these prefixes. "
        "Example: --pdfs_filter Wisconsin_Tomah Wyoming",
    )
    parser.add_argument(
        "--use_docling",
        action="store_true",
        help="Use Docling for PDF extraction and chunking instead of pypdf.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    PROMPT_VERSION = "1.0"
    prompt_version = PROMPT_VERSION

    # Load questions: skip blank lines and # comments
    questions = DEFAULT_QUESTIONS
    if args.questions_file and os.path.exists(args.questions_file):
        with open(args.questions_file) as f:
            questions = [
                l.strip() for l in f if l.strip() and not l.strip().startswith("#")
            ]
        print(f"Loaded {len(questions)} questions from {args.questions_file}")

    embed_path = _resolve_snapshot(EMBEDDING_MODELS[args.embed_model])
    print(f"Building index with embedding model: {args.embed_model}")

    if args.use_docling:
        vectorstore, site_fulltext = build_index_docling(
            args.pdfs_dir, embed_path, pdf_prefixes=args.pdfs_filter
        )
    else:
        vectorstore, site_fulltext = build_index(
            args.pdfs_dir, embed_path, pdf_prefixes=args.pdfs_filter
        )

    q_stem = (
        os.path.splitext(os.path.basename(args.questions_file))[0]
        if args.questions_file
        else "rag_results"
    )
    out_csv = os.path.join(args.out_dir, f"{q_stem}_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "llm_model",
                "prompt_version",
                "site_key",
                "question_id",
                "question",
                "answer",
                "evidence_quote",
                "evidence_source",
            ],
        )
        # writer = csv.DictWriter(f, fieldnames=["llm_model", "site_key", "question", "answer"])
        writer.writeheader()

        for model_key in args.llm_models:
            if model_key not in LLM_MODELS:
                print(f"Unknown model key '{model_key}', skipping.")
                continue

            model_path = _resolve_snapshot(LLM_MODELS[model_key])
            print(f"\nLoading LLM: {model_key}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            nl_ids = tokenizer.encode("\n", add_special_tokens=False)
            if nl_ids:
                eos_ids.extend(nl_ids)

            pipe = pipeline(
                "text-generation",
                model=model_path,
                tokenizer=tokenizer,
                max_new_tokens=256,
                repetition_penalty=1.2,
                eos_token_id=eos_ids,
                do_sample=False,
                return_full_text=False,
                device_map="auto",
                trust_remote_code=True,
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            # Run each question against each site independently
            for site_key in site_fulltext:
                for question_id, question in enumerate(questions):
                    print(f"  [{site_key}] Q: {question[:60]}")
                    answer = ask(
                        question,
                        site_key,
                        site_fulltext,
                        vectorstore,
                        llm,
                        tokenizer=tokenizer,
                    )
                    evidence_quote = ""
                    evidence_source = ""
                    # answer = ask(question, site_key, site_fulltext, vectorstore, llm)
                    print(f"  A: {answer[:100]}...")
                    writer.writerow(
                        {
                            "llm_model": model_key,
                            "prompt_version": prompt_version,
                            "site_key": site_key,
                            "question_id": question_id,
                            "question": question,
                            "answer": answer,
                            "evidence_quote": evidence_quote,
                            "evidence_source": evidence_source,
                        }
                    )
                    # writer.writerow({"llm_model": model_key, "site_key": site_key,
                    #                "question": question, "answer": answer})
                    f.flush()

            del pipe, llm

    print(f"\nDone. Results saved to: {out_csv}")


if __name__ == "__main__":
    main()

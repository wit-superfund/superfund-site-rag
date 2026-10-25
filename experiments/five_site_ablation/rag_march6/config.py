"""Configuration constants for RAG pipeline."""
import os

# Paths
BASE_DIR = "/work/pi_larsonj_wit_edu/superfund/project/superfund"
PDFS_DIR = os.path.join(BASE_DIR, "pdfs/pdfs")
SITE_INDEX_DIR = os.path.join(BASE_DIR, "site_pdf_index")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "site_indexes")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Model paths
HF_HUB = "/datasets/ai"
EMBEDDING_MODELS = {
    "e5-mistral-7b": f"{HF_HUB}/intfloat/hub/models--intfloat--e5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1",
    "qwen3-embed-8b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}
LLM_MODELS = {
    "qwen3-8b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-8B/snapshots",
    "qwen3-4b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-4B/snapshots",
    "qwen3-14b": f"{HF_HUB}/qwen3/hub/models--Qwen--Qwen3-14B/snapshots",
    "llama3-8b": f"{HF_HUB}/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots",
    "llama3-70b": f"{HF_HUB}/llama3/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots",
    "gemma4-31B": f"{HF_HUB}/gemma/hub/models--google--gemma-4-31B-it/snapshots",
    "gemma4-12B": f"{HF_HUB}/gemma/hub/models--google--gemma-4-12B-it/snapshots",
    "gemma3-12b": f"{HF_HUB}/gemma/hub/models--google--gemma-3-12b-it/snapshots",
    "gemma3-27b": f"{HF_HUB}/gemma/hub/models--google--gemma-3-27b-it/snapshots",
    "gemma2-27b": f"{HF_HUB}/gemma/hub/models--google--gemma-2-27b-it/snapshots",
    "gemma2-9b": f"{HF_HUB}/gemma/hub/models--google--gemma-2-9b-it/snapshots",
    "medgemma-27b": f"{HF_HUB}/gemma/hub/models--google--medgemma-27b-text-it/snapshots",
}

# Extraction settings. None means extract every page.
MAX_PAGES_PER_PDF = None
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 300

# Retrieval settings
TOP_K = 6

# Default questions
DEFAULT_QUESTIONS = [
    "What contaminants are present at this site?",
    "What is the location of this Superfund site?",
    "What are the health risks associated with this site?",
    "What cleanup actions have been taken?",
]

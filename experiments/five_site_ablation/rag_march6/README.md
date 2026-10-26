# RAG March 6 Module

## Current Baseline (as of March 8, 2025)

### Architecture
- **Embedding model**: `nomic-ai/nomic-embed-text-v1.5`
- **LLM**: `Gemma3-27B` (best performer: 205 avg chars, concise, accurate)
- **Index**: Per-site FAISS indexes
- **Chunking**: ~500-1000 char chunks from PDFs

### Behavior Observed
See full analysis: `/work/pi_larsonj_wit_edu/superfund/project/superfund/output/benchmark/bench_march8.md`

Gemma behaves as a **span extractor** - pulls literal text from retrieved chunks rather than reasoning across documents.

**Strengths**:
- Short, direct answers good for structured extraction
- Minimal interpretation = cleaner CSV output
- Works well for factual questions (location, contaminants list)

**Weaknesses**:
- Historical questions: retrieves investigation dates (1982) instead of site history (1890s)
- Causal questions: returns "contaminated groundwater" instead of "Hooker Chemical dumping"
- Regulatory questions: returns FYR schedule instead of NPL status

**Root cause**: Retrieval surfaces high-salience investigation docs over historical/background sections.

---

## Proposed Improvements

### Option 2: Query Expansion (SIMPLE)
Expand queries based on question type before retrieval:
```python
if "earliest" in question.lower() or "history" in question.lower():
    query = f"{question} site history background origin construction industrial activity"
```
**Effort**: ~10 lines in retrieve.py
**Expected impact**: Better historical/causal question answers

**Limitation**: Current implementation uses hardcoded trigger words ("earliest", "history", "cause", etc.). Won't generalize to arbitrary prompts like "When was the canal built?" or "Who dumped the waste?"

**Future generalization options**:
- LLM-based query expansion (ask LLM to add search terms)
- Question classification model → appropriate expansion
- HyDE (Hypothetical Document Embeddings) - generate hypothetical answer, use for retrieval

### Option 3: Two-Stage Retrieval with Cross-Encoder Re-ranking
1. Retrieve top-k chunks with FAISS (current approach)
2. Re-rank using cross-encoder that scores semantic relevance to question intent

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
**Effort**: ~30 lines, new dependency
**Expected impact**: Better ranking for complex questions

---

## Run History

| Date | Job ID | Description | Output |
|------|--------|-------------|--------|
| 2025-03-08 | 53453367 | Full evidence text (33 sites × 8 questions) | `rag_ella_gemma3-27b_*.csv` |
| 2025-03-08 | 53516804 | Gradio demo launch | (interactive) |

---

## Two-Phase Pipeline (from bench_march8 analysis)

**Phase 1 (current)**: Gemma extracts structured fields → clean CSV dataset
**Phase 2 (future)**: NotebookLM-style synthesis on thematic clusters → narrative chapters

The extraction behavior is *correct* for Phase 1. Improvements above target edge cases where extraction fails (historical, causal, regulatory questions).

# Superfund Site RAG

Research code for answering questions from EPA Superfund document archives with retrieval-augmented generation (RAG). The current tracked runner is `rag_run_july_30.py`; larger corpora, model snapshots, FAISS indexes, and experiment outputs live on the Unity cluster and are intentionally excluded from Git.

## Current direction

- Extract and index every page by default. A page cap should be used only for an explicitly labeled smoke test.
- Preserve site scope so evidence from one Superfund site cannot leak into another site's answer.
- Compare dense retrieval (`e5-mistral-7b-instruct` and `Qwen3-Embedding-8B`) with a lexical BM25 baseline and a fused BM25+dense run.
- Require answers to cite retrieved evidence and return `NOT PRESENT` when the archive does not support an answer.
- Use the five-site mining pilot in `benchmarks/five_site_benchmark.tsv` and the five reviewed questions in `benchmarks/questions_mining.txt` before expanding the evaluation set.
- Preserve baseline, one-factor-at-a-time, and incremental runs using the versioned [`benchmarks/ablation_matrix.tsv`](benchmarks/ablation_matrix.tsv) and [`benchmarks/ABLATION_PROTOCOL.md`](benchmarks/ABLATION_PROTOCOL.md).

The next repository cleanup will selectively move the active cluster modules into this repository. PDFs, model weights, indexes, virtual environments, logs, and generated answers should remain outside Git.

## Basic invocation

Activate the project environment, then run:

```bash
python rag_run_july_30.py \
  --pdfs_dir /path/to/pdfs \
  --questions_file questions.txt \
  --embed_model qwen3-embed-8b \
  --llm_models gemma4-31B-it
```

Model paths in the runner currently point to Unity's shared `/datasets/ai` snapshots and will need configuration when running elsewhere.

# Superfund Site RAG

This repository contains the code used to evaluate archive-bounded question
answering over U.S. EPA Superfund records. The current research workflow keeps
retrieval, citation resolution, answer quality, abstention, and compute cost as
separate measurements; a resolving citation is not presented as proof of
semantic entailment.

The original July runner remains at `rag_run_july_30.py`. The reproducible
five-site ablation candidate is in
[`experiments/five_site_ablation`](experiments/five_site_ablation/README.md).
It adds:

- all-page extraction by default, while retaining the historical 50-page arm;
- content-addressed extraction, chunk, FAISS, and BM25 caches;
- raw and correctly instructed Qwen3 embedding profiles;
- dense, BM25-only, and deterministic dense+BM25/RRF retrieval;
- legacy, lenient, and fail-closed citation-contract evaluation;
- immutable arm manifests and retrieval/generation/verification timing; and
- an explicit 13-arm individual-plus-incremental experiment matrix.

The exact `Qwen/Qwen3-Reranker-0.6B` snapshot is not currently installed on the
Unity cluster. `I_rerank` and `L5` therefore remain intentionally blocked; the
installed generic Qwen3 language models are not substituted.

### Local checks

The integration contract tests do not load model weights or require a GPU:

```bash
cd experiments/five_site_ablation
PYTHONPATH=. python -m unittest discover -s tests -v
bash -n run_ablation_arm.slurm
```

The architecture review, experiment rationale, and “what needs fixing first”
plan are in
[`docs/rag_architecture_review_2026-10-22.md`](docs/rag_architecture_review_2026-10-22.md).

Large source PDFs, extracted corpora, model snapshots, indexes, logs, and answer
outputs are intentionally excluded from Git. Run manifests record hashes and
configuration needed to associate those external artifacts with a code version.

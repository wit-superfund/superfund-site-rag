# Integrated RAG ablation candidate

This directory combines the four low-complexity streams without changing the
active Unity pipeline:

- content-addressed extraction, chunk, FAISS, and BM25 caches;
- raw or correctly instructed Qwen3 query embeddings (documents stay raw);
- dense, BM25-only, or deterministic dense+BM25/RRF retrieval; and
- legacy, lenient, or strict citation-contract verification.

Qwen3 runs fail closed unless the checkpoint contains the official tokenizer,
Sentence Transformers module graph, last-token pooler, and normalization
module. Unity's shared weight snapshot omits those small interface files; using
it directly produced the same vector for every query. The checked-in
`prepare_qwen3_embedding_overlay.py` creates a version-pinned complete view by
symlinking the shared 8B weights and verifying separately downloaded interface
assets from revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. The runner records the
validated interface-asset bundle hash in every Qwen index and result.

`compare_rag.py` defaults to **all pages** (`--max_pages 0`). The historical
50-page behavior is retained only when an arm explicitly selects 50. BM25-only
requires `--embed_model none` and therefore skips loading embeddings and
building FAISS. Dense/hybrid runs require a real embedding model.

The CSV retains the unmodified model output in `raw_response` and the existing
cleaned output in `answer` for audit and human evaluation. Strict verification
writes the user-visible result to `released_answer`:
valid answers pass through, exact `NOT PRESENT` remains an abstention, and an
invalid citation contract is suppressed as `INSUFFICIENT_EVIDENCE` with
`answer_disposition=rejected`. This is a syntactic/provenance gate, not a
semantic entailment check; `semantic_support_checked` remains false.

## Matrix contract

`ablation_matrix.tsv` is the single arm registry. Individual arms change one
factor from `B0`, so Qwen, BM25/hybrid, and verifier-only arms retain the
50-page cap. The ladder changes to all pages at `L1`:

`questions_mining.tsv` is the canonical five-question input and assigns stable
semantic IDs. Legacy one-question-per-line files remain accepted, but receive
explicit positional IDs and should not be used for new paper runs.

- `L1` is identical to `I_pages` and must reuse that result;
- `L2` adds instructed Qwen3;
- `L3` adds BM25/RRF; and
- `L4` adds strict citation verification.

`I_rerank` and `L5` deliberately fail closed because the exact
`Qwen/Qwen3-Reranker-0.6B` snapshot is not installed on Unity. A generic
Qwen3-0.6B language model is not a substitute.

Every executable arm writes a configuration manifest, answer rows, per-source
timing rows, and GNU `time -v` resource data. Rows include cache keys/hits,
retrieval mode and score semantics, query-instruction identity, citation status
and reason codes, retrieval/generation/verification/total question timing, and
peak CUDA allocation/reservation. The runner refuses to mix a different
configuration into an existing run directory.

## Validation and rollout

Login-node-safe checks:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
bash -n run_ablation_arm.slurm
```

The repository form includes 79 pure-Python contract/regression tests. They
include stable question-ID parsing, and one of them binds the smoke default to
EPA ID `MTD980502777` so a site-file reorder
cannot silently redirect a Silver Bow smoke to another site.

The completed October 25 Silver Bow smoke audit, including final Slurm job IDs,
cache timings, and caveats, is recorded in
[`SMOKE_RESULTS_2026-10-25.md`](SMOKE_RESULTS_2026-10-25.md).
The resulting five-site submission graph and full-run job IDs are recorded in
[`FULL_RUN_2026-10-25.md`](FULL_RUN_2026-10-25.md).
The completed output-integrity and preliminary behavior audit is recorded in
[`RESULTS_AUDIT_2026-10-25.md`](RESULTS_AUDIT_2026-10-25.md).
The Qwen/PDF defect investigation, corrected pipeline, direct probe evidence,
and correction-run gates are recorded in
[`CORRECTION_RUN_2026-10-25.md`](CORRECTION_RUN_2026-10-25.md).

## Qwen3 checkpoint preparation

Download these six non-weight files from the exact frozen
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af)
revision into one directory: `modules.json`,
`config_sentence_transformers.json`, `1_Pooling/config.json` (rename this one
to `pooling_config.json` in the asset directory), `tokenizer.json`,
`merges.txt`, and `vocab.json`. Then run:

```bash
python prepare_qwen3_embedding_overlay.py \
  --source /datasets/ai/qwen3/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --destination /work/pi_larsonj_wit_edu/superfund/project/superfund/models/qwen3-embedding-8b-1d8ad4ca-complete \
  --assets /path/to/downloaded/assets
```

The builder verifies all six SHA-256 hashes and refuses to replace conflicting
files. `compare_rag.py` independently checks the module graph, tokenizer,
last-token pooling configuration, and records the resulting bundle hash.

One-question, three-page Silver Bow smoke for an executable arm:

```bash
sbatch --array=1 --export=ALL,ARM=L4,SMOKE=1 run_ablation_arm.slurm
```

Full five-site submission should happen only after dense, instructed-Qwen,
BM25-only, and hybrid+strict smokes complete and their manifests/output columns
are inspected. Example:

```bash
sbatch --array=0-4 --export=ALL,ARM=I_bm25 run_ablation_arm.slurm
```

The completed first-pass matrix remains staged at
`/work/pi_larsonj_wit_edu/superfund/project/superfund/staging/rag_matrix_v2_20261025`.
The corrected PDF-fallback pipeline is staged at
`/work/pi_larsonj_wit_edu/superfund/project/superfund/staging/rag_matrix_v3_20261025`
and writes to `output/rag_matrix_v3_pdf_fallback`. It intentionally reuses the
content-addressed v2 cache: unchanged condensed artifacts remain valid, while
the new PDF extraction schema forces PDF extraction, chunks, and indexes to be
rebuilt. PDF timing rows report fallback, empty-page, and extraction-failure
counts and document IDs.
The active `rag_condensed_compare` directory is intentionally untouched.

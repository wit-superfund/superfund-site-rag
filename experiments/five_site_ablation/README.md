# Integrated RAG ablation candidate

This directory combines the four low-complexity streams without changing the
active Unity pipeline:

- content-addressed extraction, chunk, FAISS, and BM25 caches;
- raw or correctly instructed Qwen3 query embeddings (documents stay raw);
- dense, BM25-only, or deterministic dense+BM25/RRF retrieval; and
- legacy, lenient, or strict citation-contract verification.

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

The repository form includes 71 pure-Python contract/regression tests. They
include stable question-ID parsing, and one of them binds the smoke default to
EPA ID `MTD980502777` so a site-file reorder
cannot silently redirect a Silver Bow smoke to another site.

The completed October 25 Silver Bow smoke audit, including final Slurm job IDs,
cache timings, and caveats, is recorded in
[`SMOKE_RESULTS_2026-10-25.md`](SMOKE_RESULTS_2026-10-25.md).

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

The current Unity staging location is
`/work/pi_larsonj_wit_edu/superfund/project/superfund/staging/rag_matrix_v2_20261025`.
The active `rag_condensed_compare` directory is intentionally untouched.

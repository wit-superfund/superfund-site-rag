# Five-site ablation protocol

## Goal

Measure the quality and compute effect of each RAG change on the same five sites and five reviewed questions. Keep every raw run so the selected system can omit changes that do not provide a useful improvement.

## Two experiment series

1. **Individual arms (`I_*`)** change one factor from `B0`. These estimate the effect of that factor without attributing gains from another change to it.
2. **Incremental arms (`L*`)** add accepted changes in deployment order. These detect interactions between changes and measure the final combined system.

`I_pages` and `L1` are the same configuration, so one result may be referenced by both labels. A calibration arm such as `I_qwen_raw` is diagnostic and should not be treated as the correctly configured Qwen3 result.

## Frozen inputs

- the five rows in `five_site_benchmark.tsv`;
- the five questions in `questions_mining.txt`, with stable IDs assigned by the runner;
- one immutable corpus snapshot or content manifest;
- identical chunk size, overlap, site filtering, generator, prompt, decoding, and top-k except where the matrix explicitly changes them; and
- exact model snapshot identifiers rather than mutable model names.

The baseline retains the historical 50-page cap so the all-pages change can be measured. The proposed final ladder uses all pages.

## Required output for every run

Store results under a unique `run_id` without overwriting earlier runs:

- `run_manifest`: timestamps, host/job, code commit or code hashes, corpus/questions/site-manifest hashes, and every configuration value;
- `answers`: site, question ID, answer, abstention status, citations, and verification result;
- `retrieval`: every candidate's site, document, page, chunk ID, component ranks/scores, fused rank, reranker score, and selected-context flag;
- `timing`: extraction, embedding/indexing, retrieval, reranking, verification, generation, and total wall time;
- `resources`: maximum resident memory, peak GPU memory when available, index size, and chunk/vector counts; and
- `errors`: extraction failures, empty pages, model failures, and retries.

The current cluster benchmark already records extraction, embedding, retrieval, generation, document/page/chunk counts, and model-loading time. The ablation launcher adds a per-site configuration manifest and GNU `time -v` output for wall time and maximum resident memory. Peak GPU memory and full candidate traces remain implementation items.

## Metrics

Use retrieval metrics before judging generated prose:

- Recall@5/10/20, MRR, nDCG, and complete-evidence-set recall where gold pages exist;
- wrong-site leakage and citation/file/page/quote validity;
- answer grade and abstention precision/recall;
- indexing time, query latency, generation time, total wall time, index bytes, CPU memory, and GPU memory; and
- quality deltas paired by the same site/question, with uncertainty reported because the pilot is small.

The 15 site-specific questions with expected answers and source pages for Anaconda, Silver Bow/Butte, and Milltown are the strongest current retrieval test. The shared five questions across all five sites provide broader end-to-end coverage but need equivalent evidence labels for California Gulch and Summitville.

## Selection rule

Choose the primary quality metric and practical compute limits before examining the final results. Retain a change only when its paired improvement is meaningful for the target metric or it fixes an important safety failure without violating the latency/memory limits. Report negative and null ablations rather than deleting them. “Revert” means excluding the change from the selected configuration; its code, manifest, and results remain available for reproducibility.

Use deterministic decoding for the main comparison. Warm models before timing, repeat short retrieval-only timings, and report medians and tail latency. Never reuse an index unless its manifest matches the corpus, page policy, chunker, embedding snapshot, instruction, and dimensionality.

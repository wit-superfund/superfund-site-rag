# Silver Bow smoke validation — 2026-10-25

The final pre-benchmark smoke chain completed successfully on Unity's NVIDIA
H200 NVL node. It used Silver Bow Creek/Butte Area (`MTD980502777`), the
`mining_history` question, the condensed source, and at most three pages per
PDF. The resulting corpus contained 728 documents, 1,970 pages, and 1,815
chunks.

This is a pipeline and cache validation, not an answer-quality result. The
three-page cap is smoke-only; executable paper arms retain their matrix-defined
50-page or all-page settings. All five smoke runs returned the exact abstention
`NOT PRESENT`, which is valid behavior when this deliberately truncated corpus
does not provide enough evidence.

| Check | Slurm job | Elapsed | Index behavior | Peak GPU allocated |
|---|---:|---:|---|---:|
| E5 dense, cold cache (`I_pages`) | `66047163_1` | 1:40 | FAISS built in 57.39 s | 79.290 GB |
| E5 dense, warm cache (`I_pages`) | `66047164_1` | 0:41 | extraction/chunks/FAISS hit; FAISS loaded in 0.04 s | 79.289 GB |
| Instructed Qwen3 dense (`I_qwen`) | `66085422_1` | 1:23 | extraction/chunks/FAISS hit; FAISS loaded in 0.08 s | 78.164 GB |
| BM25 only (`I_bm25`) | `66085423_1` | 0:33 | no embedding model or FAISS; BM25 built in 0.20 s | 66.902 GB |
| Qwen3 + BM25/RRF + strict verification (`L4`) | `66085424_1` | 0:24 | Qwen FAISS hit in 0.02 s; BM25 hit in 0.01 s | 79.985 GB |

The Qwen smoke initially exposed a LangChain compatibility defect: a loaded
FAISS index treated the query-aware embedding adapter as a callable. The
adapter now implements that compatibility path and, when available, subclasses
LangChain's embedding interface. A regression test confirms that both paths
apply the instruction to queries only; document embeddings remain unchanged.

The final L4 answer row also confirms the fail-closed output contract:
`verifier_mode=strict`, `citation_status=abstained`,
`answer_disposition=abstained`, and `released_answer=NOT PRESENT`. Its manifest
records the instructed-Qwen hash, both cache keys, hybrid RRF score semantics,
and the stable `mining_history` question ID.

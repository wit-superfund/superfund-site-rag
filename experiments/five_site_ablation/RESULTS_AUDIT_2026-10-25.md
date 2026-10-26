# Five-site result audit — 2026-10-25

All 10 executable arms completed for five sites and two source variants. The
100 answer CSVs contain exactly 500 rows. Every arm has 50 rows, five distinct
site IDs, both `pdf` and `condensed` sources, and all five stable question IDs.
No task failed and no completed log contains a traceback or exception.

These are behavioral and runtime summaries, not answer-accuracy scores. The
answers still need evaluation against the verified reference answers before
any retrieval method is ranked or retained. The Qwen arms and raw-PDF arms in
this first-pass matrix have subsequently failed implementation/coverage audits
described below and must not be used for comparative conclusions.

| Arm | Accepted | Abstained | Rejected | Median retrieval | Median generation | Median question total |
|---|---:|---:|---:|---:|---:|---:|
| `B0` | 26 | 24 | 0 | 38.55 ms | 2.395 s | 2.470 s |
| `I_pages` | 31 | 19 | 0 | 57.65 ms | 2.935 s | 3.031 s |
| `I_qwen_raw` | 19 | 31 | 0 | 21.05 ms | 0.935 s | 0.958 s |
| `I_qwen` | 19 | 31 | 0 | 22.95 ms | 0.935 s | 0.958 s |
| `I_bm25` | 38 | 12 | 0 | 4.40 ms | 3.470 s | 3.479 s |
| `I_hybrid` | 34 | 16 | 0 | 44.70 ms | 3.220 s | 3.285 s |
| `I_verify` | 25 | 24 | 1 | 36.10 ms | 2.375 s | 2.394 s |
| `L2` | 21 | 29 | 0 | 33.10 ms | 0.940 s | 0.994 s |
| `L3` | 36 | 14 | 0 | 50.15 ms | 2.815 s | 2.894 s |
| `L4` | 34 | 14 | 2 | 53.05 ms | 2.845 s | 2.915 s |

The strict verifier rejected three mixed-abstention responses and released
`INSUFFICIENT_EVIDENCE` instead: one `I_verify` row and two `L4` rows. The other
strict rows were either exact abstentions or passed the citation contract.

## Sanity checks required before interpretation

- `I_qwen_raw`, `I_qwen`, `L2`, `L3`, and `L4` are invalid in this first-pass
  output. Direct GPU probe `66132586` showed that all five raw questions and
  all five instructed questions produced the exact same 4,096-dimensional
  vector (the same SHA-256 for every vector). Unity's local Qwen weight
  snapshot lacked `tokenizer.json`, `modules.json`, the last-token pooling
  configuration, and other official embedding interface assets. The pipeline
  now fails closed on that incomplete snapshot and uses a version-pinned
  complete overlay for the correction run. Corrected direct probe `66132674`
  produced a distinct SHA-256 for every raw question and for every instructed
  counterpart; raw/instructed cosine similarity ranged from 0.649 to 0.858.
- Raw `pypdf` extraction skipped `08100008059.pdf` and `08100008111.pdf`
  because the Unity environment lacks the optional `cryptography` package.
  PyMuPDF can recover 351,381 characters from the first file; the second yields
  only 20 characters and likely needs OCR. Neither file appears in the
  condensed corpus. Extraction coverage must be fixed or explicitly reported
  before the raw-versus-condensed comparison is paper-ready.
- Accepted, abstained, and citation-valid counts measure pipeline behavior,
  not factual correctness or semantic support. Gold-answer scoring and human
  citation review remain necessary.

Accordingly, the table above is retained as an audit trail of the first run,
not as a method ranking. The correction matrix must finish before effect sizes
are calculated.

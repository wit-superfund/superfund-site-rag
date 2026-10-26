# Five-site correction run — 2026-10-25

The first complete matrix is retained as an audit artifact, but two issues
must be corrected before comparing arms:

1. Raw PDF extraction silently omitted two encrypted Silver Bow files when
   `pypdf` could not open them without the optional cryptography package.
2. Unity's Qwen3 weight snapshot omitted the official tokenizer and Sentence
   Transformers embedding interface. It silently produced the same vector for
   every query.

## Implemented corrections

- PDF extraction now uses `pypdf` first and PyMuPDF on document/page parser
  errors, preserves original page numbers, stores diagnostics in the cached
  extraction artifact, and exposes fallback/failure/empty-page fields in the
  timing CSV. The new PDF extraction schema invalidates only PDF-derived
  extraction, chunks, and indexes; unchanged condensed artifacts remain
  reusable.
- Qwen3 now uses a version-pinned overlay for revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. It symlinks the shared 8B
  weights and adds hash-verified official tokenizer, module graph, last-token
  pooler, and normalization assets. Runtime validation fails closed if those
  assets are absent or inconsistent. Qwen index keys include the validated
  interface bundle hash.

The new stage is
`/work/pi_larsonj_wit_edu/superfund/project/superfund/staging/rag_matrix_v3_20261025`;
outputs go to `output/rag_matrix_v3_pdf_fallback`. It reuses the
content-addressed `rag_matrix_v2_cache/v1` cache only when the full artifact
configuration still matches.

## Validation evidence

- Local pure-Python suite: 79/79 passed.
- Frozen Unity stage suite: 79/79 passed.
- Direct PDF fallback check:
  - `08100008059.pdf`: 139 pages, 350,999 extracted characters, no parser-failed pages.
  - `08100008111.pdf`: 50 pages, 19 extracted characters, no parser-failed pages;
    this file remains an OCR candidate.
- Invalid Qwen probe `66132586`: every raw/instructed question had the same
  4,096-dimensional vector and SHA-256.
- Corrected Qwen probe `66132674`: all raw questions had distinct SHA-256s;
  all instructed vectors differed from raw, with raw/instructed cosine
  similarity 0.649–0.858. The validated interface bundle is
  `bf5ce1dd37094feb5003d77394a3ea3fe2834cef3a1f7b07bd0b2a2bb5f57314`.

Raw JSONL probe logs are archived locally under
`results/rag_matrix_v2_20261025/probes/`.

## Slurm gates

| Job | Purpose | Status |
|---|---|---|
| `66132606_1` | Silver Bow, PDF, `I_pages` fallback smoke | queued |
| `66132687_1` | Silver Bow, PDF, corrected `I_qwen` smoke | queued |
| `66132688_1` | Silver Bow, PDF, corrected `L4` hybrid+strict smoke | queued |

The ten-arm correction matrix should be submitted only after these three
smokes produce complete manifests, answer/timing rows, expected extraction
diagnostics, and no traceback.

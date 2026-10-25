#!/usr/bin/env python3
"""Post-hoc citation-contract evaluation for saved answer CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from citation_eval import verify_text


@dataclass(frozen=True)
class SavedChunk:
    pdf_filename: str
    page: int


_SAVED_SOURCE = re.compile(r"^\s*(?P<filename>.+?):p(?P<page>\d+)\s*$", re.IGNORECASE)


def parse_retrieved_sources(value: str) -> List[SavedChunk]:
    chunks = []
    for item in (value or "").split("|"):
        match = _SAVED_SOURCE.fullmatch(item.strip())
        if match:
            chunks.append(
                SavedChunk(
                    pdf_filename=match.group("filename").strip(),
                    page=int(match.group("page")),
                )
            )
    return chunks


EVAL_FIELDS = [
    "eval_mode",
    "eval_status",
    "eval_answer_valid",
    "eval_citation_applicable",
    "eval_citation_verified",
    "eval_abstained",
    "eval_citations_parsed",
    "eval_citations_verified",
    "eval_citations_failed",
    "eval_reason_codes",
    "eval_verified_citations",
    "eval_failed_citations",
    "eval_parse_issues",
    "eval_evidence_scope",
    "eval_semantic_support_checked",
]


def evaluate_row(row: Dict[str, str], *, mode: str) -> Dict[str, str]:
    chunks = parse_retrieved_sources(row.get("retrieved_sources", ""))
    result = verify_text(row.get("answer", ""), chunks, mode=mode)
    applicable = not result.is_abstention
    citation_value = (
        "not_applicable"
        if result.citation_verified is None
        else str(result.citation_verified).lower()
    )
    scope = "retrieved_source_metadata" if chunks else "none"
    reasons = list(result.reason_codes)
    if applicable and result.citation_count and not chunks:
        reasons.append("posthoc_evidence_missing")

    additions = {
        "eval_mode": mode,
        "eval_status": result.status,
        "eval_answer_valid": str(result.is_valid).lower(),
        "eval_citation_applicable": str(applicable).lower(),
        "eval_citation_verified": citation_value,
        "eval_abstained": str(result.is_abstention).lower(),
        "eval_citations_parsed": str(result.citation_count),
        "eval_citations_verified": str(len(result.verified_sources)),
        "eval_citations_failed": str(len(result.failed_citations)),
        "eval_reason_codes": "; ".join(dict.fromkeys(reasons)),
        "eval_verified_citations": "; ".join(result.verified_sources),
        "eval_failed_citations": "; ".join(result.failed_citations),
        "eval_parse_issues": "; ".join(result.parse_issues),
        "eval_evidence_scope": scope,
        "eval_semantic_support_checked": "false",
    }
    return {**row, **additions}


def reevaluate_csv(input_path: str, output_path: str, *, mode: str = "strict") -> int:
    with open(input_path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "answer" not in reader.fieldnames:
            raise ValueError("input CSV must contain an 'answer' column")
        fieldnames = list(reader.fieldnames)
        for field in EVAL_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)
        rows = [evaluate_row(row, mode=mode) for row in reader]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--mode", choices=["lenient", "strict"], default="strict")
    args = parser.parse_args(argv)
    count = reevaluate_csv(args.input_csv, args.output_csv, mode=args.mode)
    print(f"evaluated {count} rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

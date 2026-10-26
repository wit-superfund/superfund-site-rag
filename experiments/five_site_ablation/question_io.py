"""Versioned question-file loading for benchmark runs."""

from __future__ import annotations

from pathlib import Path


def load_questions(path: str | Path) -> list[tuple[str, str]]:
    """Load ``(question_id, question)`` pairs.

    Preferred input is a tab-separated ID and question on each non-comment
    line. Legacy one-question-per-line files remain supported with positional
    IDs ``q0``, ``q1``, and so on.
    """
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                question_id, question = (part.strip() for part in line.split("\t", 1))
            else:
                question_id, question = f"q{len(records)}", line
            if not question_id or not question:
                raise ValueError(f"invalid question record on line {line_number}")
            if question_id in seen:
                raise ValueError(f"duplicate question_id {question_id!r} on line {line_number}")
            seen.add(question_id)
            records.append((question_id, question))
    if not records:
        raise ValueError("questions file contains no questions")
    return records

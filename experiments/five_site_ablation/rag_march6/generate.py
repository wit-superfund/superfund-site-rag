"""LLM generation with citation requirements."""
from dataclasses import dataclass
from typing import List, Optional
import json
import re

from .retrieve import RetrievedChunk, format_context


def strip_cot(text: str) -> str:
    """Strip chain-of-thought reasoning, keep only final answer."""
    # Remove <think>...</think> blocks (Qwen3 native format)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Cut at reasoning markers (fallback for untagged CoT)
    for marker in ["\nWait,", "\nLet me", "\nLooking", "\nOkay,", "\nHmm", "\nThe answer is", "\nThis is because", "\nI "]:
        if marker in text:
            text = text.split(marker)[0]
    # If the entire answer is NOT PRESENT, just return that
    if text.strip().startswith("NOT PRESENT"):
        return "NOT PRESENT"
    # Strip trailing "NOT PRESENT" when the model already gave cited content
    text = re.sub(r'\s*\.?\s*NOT PRESENT\s*$', '', text.strip())
    # Remove repeated sentences
    lines = text.strip().split("\n")
    seen = set()
    unique = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            unique.append(line)
    return "\n".join(unique).strip()


@dataclass
class Answer:
    """Generated answer with evidence."""
    text: str
    evidence_quote: str
    evidence_source: str
    raw_response: str


def build_prompt(question: str, chunks: List[RetrievedChunk], site_key: str) -> str:
    """Build prompt with context and citation instructions."""
    context = format_context(chunks)
    site_name = site_key.replace("_", " ")

    return f"""Extract information from EPA Superfund documents about the {site_name} site. Rules:
1. Use ONLY the sources below. Do not add outside knowledge.
2. Cite every fact as [Source N].
3. Focus your answer on the {site_name} site specifically, not on other sites mentioned in the documents.
4. If the answer is not in the sources at all, say exactly: NOT PRESENT (do not combine NOT PRESENT with other content).
5. Be brief (1-3 sentences max).

SOURCES:
{context}

Question about {site_name}: {question}
Answer:"""


def generate(llm, question: str, chunks: List[RetrievedChunk], site_key: str) -> Answer:
    """Generate answer using LLM with retrieved context.

    Args:
        llm: LangChain LLM instance
        question: User question
        chunks: Retrieved chunks with source info
        site_key: Site identifier

    Returns: Answer with evidence
    """
    if not chunks:
        return Answer(
            text="No relevant information found in the PDFs for this site.",
            evidence_quote="",
            evidence_source="",
            raw_response="",
        )

    prompt = build_prompt(question, chunks, site_key)
    raw = llm.invoke(prompt)

    # Clean response (strip prompt echo if present)
    text = raw
    # Check for prompt echo - split on "Answer:" marker
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    # Strip CoT reasoning
    text = strip_cot(text)

    # Get evidence from top chunks
    evidence_quotes = []
    evidence_sources = []
    for c in chunks[:3]:
        quote = c.text.replace("\n", " ").strip()
        evidence_quotes.append(quote)
        evidence_sources.append(f"{c.pdf_filename}:p{c.page}")

    return Answer(
        text=text,
        evidence_quote=" | ".join(evidence_quotes),
        evidence_source=" | ".join(evidence_sources),
        raw_response=raw,
    )

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


INTEGRATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_DIR))

from bm25_rrf import BM25Index  # noqa: E402
from citation_eval import verify_text  # noqa: E402
from embedding_modes import QueryAwareEmbeddings, build_embedding_profile  # noqa: E402
from retrieval_modes import RetrievalSettings, get_or_build_bm25, retrieve_question  # noqa: E402


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    pdf_filename: str
    page: int
    epa_url: str = ""
    score: float = 0.0


class SpyEmbeddings:
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0]


class IntegrationContractTests(unittest.TestCase):
    def test_qwen_instruction_is_query_only(self):
        delegate = SpyEmbeddings()
        profile = build_embedding_profile("qwen3-embed-8b", "superfund")
        wrapped = QueryAwareEmbeddings(delegate, profile)
        wrapped.embed_documents(["document text"])
        wrapped.embed_query("Where was mining performed?")
        self.assertEqual(delegate.documents, [["document text"]])
        self.assertTrue(delegate.queries[0].startswith("Instruct: "))
        self.assertIn("\nQuery:Where was mining performed?", delegate.queries[0])

    def test_profile_identity_separates_raw_and_instructed_qwen(self):
        raw = build_embedding_profile("qwen3-embed-8b", "none")
        instructed = build_embedding_profile("qwen3-embed-8b", "superfund")
        self.assertNotEqual(raw.index_key, instructed.index_key)
        self.assertNotEqual(raw.instruction_sha256, instructed.instruction_sha256)

    def test_bm25_cache_reuses_exact_chunk_configuration(self):
        documents = [
            Document("copper tailings", {"pdf_filename": "one.pdf", "page": 1}),
            Document("arsenic groundwater", {"pdf_filename": "two.pdf", "page": 2}),
        ]
        settings = RetrievalSettings(mode="bm25", candidate_k=6, top_k=2)
        with tempfile.TemporaryDirectory() as directory:
            first, first_event = get_or_build_bm25(directory, "chunks-key", documents, settings.bm25_config)
            second, second_event = get_or_build_bm25(directory, "chunks-key", documents, settings.bm25_config)
        self.assertIsInstance(first, BM25Index)
        self.assertEqual(first.document_count, second.document_count)
        self.assertFalse(first_event.hit)
        self.assertTrue(second_event.hit)

    def test_hybrid_deduplicates_and_preserves_final_top_k(self):
        documents = [
            Document("copper tailings", {"pdf_filename": "shared.pdf", "page": 1}),
            Document("arsenic groundwater", {"pdf_filename": "lexical.pdf", "page": 2}),
        ]
        bm25 = BM25Index.build(doc.page_content for doc in documents)
        settings = RetrievalSettings(mode="hybrid", candidate_k=2, top_k=2)

        def dense_retrieve(_index, _question, top_k):
            self.assertEqual(top_k, 2)
            return [Chunk("copper tailings", "shared.pdf", 1, score=0.1)]

        got = retrieve_question(
            settings,
            question="copper arsenic",
            chunks=documents,
            retrieved_chunk_type=Chunk,
            dense_index=object(),
            bm25_index=bm25,
            dense_retrieve=dense_retrieve,
        )
        identities = [(item.pdf_filename, item.page, item.text) for item in got]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertLessEqual(len(got), settings.top_k)

    def test_strict_verifier_is_fail_closed_and_abstention_is_tri_state(self):
        chunks = [Chunk("evidence", "report.pdf", 7)]
        uncited = verify_text("The site contained arsenic.", chunks, mode="strict")
        abstained = verify_text("NOT PRESENT", chunks, mode="strict")
        cited = verify_text("The site contained arsenic [Source 1, p.7].", chunks, mode="strict")
        self.assertFalse(uncited.is_valid)
        self.assertEqual(uncited.status, "uncited")
        self.assertTrue(abstained.is_valid)
        self.assertIsNone(abstained.citation_verified)
        self.assertTrue(cited.is_valid)
        self.assertTrue(cited.citation_verified)


if __name__ == "__main__":
    unittest.main()

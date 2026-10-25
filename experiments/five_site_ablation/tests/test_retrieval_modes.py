from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


STREAM_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = STREAM_DIR.parents[1]
sys.path[:0] = [str(STREAM_DIR / "candidate"), str(STREAM_DIR), str(ROOT_DIR / "streams" / "cache")]

from bm25_rrf import BM25Index  # noqa: E402
from retrieval_modes import (  # noqa: E402
    RetrievalSettings,
    get_or_build_bm25,
    retrieve_question,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FakeChunk:
    text: str
    pdf_filename: str
    page: int
    epa_url: str
    score: float


def retrieved(name: str, score: float = 0.0) -> FakeChunk:
    return FakeChunk(name, f"{name}.pdf", 1, "", score)


class RetrievalModeTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            FakeDocument("dense lexical shared", {"pdf_filename": "shared.pdf", "page": 1}),
            FakeDocument("lexical copper", {"pdf_filename": "lexical.pdf", "page": 2}),
        ]
        self.bm25 = BM25Index.build(doc.page_content for doc in self.documents)

    def test_dense_uses_final_top_k_not_candidate_depth(self):
        calls = []

        def dense_retrieve(index, question, top_k):
            calls.append(top_k)
            return [retrieved("dense")]

        settings = RetrievalSettings(mode="dense", candidate_k=30, top_k=6)
        result = retrieve_question(
            settings,
            question="anything",
            chunks=self.documents,
            retrieved_chunk_type=FakeChunk,
            dense_index=object(),
            dense_retrieve=dense_retrieve,
        )
        self.assertEqual(calls, [6])
        self.assertEqual(result[0].pdf_filename, "dense.pdf")

    def test_bm25_needs_no_dense_index(self):
        settings = RetrievalSettings(mode="bm25", candidate_k=2, top_k=1)
        result = retrieve_question(
            settings,
            question="copper",
            chunks=self.documents,
            retrieved_chunk_type=FakeChunk,
            bm25_index=self.bm25,
        )
        self.assertEqual(result[0].pdf_filename, "lexical.pdf")
        self.assertEqual(settings.score_semantics, "bm25-score-higher-is-better")

    def test_hybrid_fetches_candidate_depth_from_both(self):
        calls = []

        def dense_retrieve(index, question, top_k):
            calls.append(top_k)
            return [retrieved("dense")]

        settings = RetrievalSettings(mode="hybrid", candidate_k=2, top_k=1, rrf_k=60)
        result = retrieve_question(
            settings,
            question="copper",
            chunks=self.documents,
            retrieved_chunk_type=FakeChunk,
            dense_index=object(),
            bm25_index=self.bm25,
            dense_retrieve=dense_retrieve,
        )
        self.assertEqual(calls, [2])
        self.assertEqual(len(result), 1)
        self.assertEqual(settings.score_semantics, "rrf-score-higher-is-better")

    def test_content_addressed_bm25_cache_hits_on_second_build(self):
        settings = RetrievalSettings(mode="bm25", candidate_k=2, top_k=1)
        with tempfile.TemporaryDirectory() as cache_dir:
            first, first_event = get_or_build_bm25(
                cache_dir, "chunk-key-a", self.documents, settings.bm25_config
            )
            second, second_event = get_or_build_bm25(
                cache_dir, "chunk-key-a", self.documents, settings.bm25_config
            )
        self.assertFalse(first_event.hit)
        self.assertTrue(second_event.hit)
        self.assertEqual(first.search("copper", 2), second.search("copper", 2))

    def test_cache_key_changes_with_chunk_key_and_bm25_parameters(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _, base_event = get_or_build_bm25(
                cache_dir,
                "chunk-key-a",
                self.documents,
                RetrievalSettings(mode="bm25", candidate_k=2, top_k=1).bm25_config,
            )
            _, chunk_event = get_or_build_bm25(
                cache_dir,
                "chunk-key-b",
                self.documents,
                RetrievalSettings(mode="bm25", candidate_k=2, top_k=1).bm25_config,
            )
            _, parameter_event = get_or_build_bm25(
                cache_dir,
                "chunk-key-a",
                self.documents,
                RetrievalSettings(mode="bm25", candidate_k=2, top_k=1, bm25_k1=1.5).bm25_config,
            )
        self.assertNotEqual(base_event.key, chunk_event.key)
        self.assertNotEqual(base_event.key, parameter_event.key)


if __name__ == "__main__":
    unittest.main()

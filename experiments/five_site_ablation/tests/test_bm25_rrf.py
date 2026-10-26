from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


STREAM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM_DIR))

from bm25_rrf import (  # noqa: E402
    BM25Config,
    BM25Index,
    TOKENIZER_UNICODE_ALNUM_V1,
    bm25_cache_config,
    bm25_retrieve,
    reciprocal_rank_fusion,
    retrieval_config,
    tokenize,
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


def chunk(name: str, page: int, text: str | None = None, score: float = 0.0) -> FakeChunk:
    return FakeChunk(text or name, name, page, f"https://epa.example/{name}", score)


class TokenizerTests(unittest.TestCase):
    def test_unicode_casefold_ids_and_noise_policy(self):
        huge = "x" * 65
        self.assertEqual(
            tokenize(f"CAFÉ _ignored_ COD980717938 a {huge}"),
            ["café", "ignored", "cod980717938"],
        )

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError):
            tokenize("mine waste", "future-policy")


class BM25Tests(unittest.TestCase):
    def test_ranking_prefers_more_term_evidence(self):
        index = BM25Index.build([
            "arsenic tailings cleanup",
            "arsenic arsenic arsenic tailings cleanup",
            "unrelated river ecology",
        ])
        hits = index.search("arsenic tailings", 3)
        self.assertEqual([hit.document_index for hit in hits], [1, 0])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_ties_use_document_order(self):
        index = BM25Index.build(["copper", "copper", "copper"])
        self.assertEqual([hit.document_index for hit in index.search("copper", 2)], [0, 1])

    def test_query_repetition_does_not_change_scores(self):
        index = BM25Index.build(["copper mine", "mine waste"])
        once = index.search("copper", 2)
        repeated = index.search("copper copper copper", 2)
        self.assertEqual(once, repeated)

    def test_empty_and_out_of_vocabulary_queries(self):
        index = BM25Index.build(["copper mine"])
        self.assertEqual(index.search("", 5), [])
        self.assertEqual(index.search("a _", 5), [])
        self.assertEqual(index.search("pregnancy", 5), [])

    def test_empty_corpus(self):
        self.assertEqual(BM25Index.build([]).search("copper", 5), [])

    def test_persistence_round_trip(self):
        original = BM25Index.build(["copper mine", "mine tailings"])
        with tempfile.TemporaryDirectory() as directory:
            original.save(directory)
            loaded = BM25Index.load(directory)
        self.assertEqual(original.config, loaded.config)
        self.assertEqual(original.search("mine", 2), loaded.search("mine", 2))

    def test_retrieved_chunk_preserves_file_page_url_and_text(self):
        documents = [
            FakeDocument("arsenic waste", {
                "pdf_filename": "record.pdf",
                "page": 37,
                "epa_url": "https://epa.example/record",
                "chunk_index": 4,
            })
        ]
        index = BM25Index.build(doc.page_content for doc in documents)
        result = bm25_retrieve(index, documents, "arsenic", top_k=1, chunk_type=FakeChunk)
        self.assertEqual(result[0].text, "arsenic waste")
        self.assertEqual(result[0].pdf_filename, "record.pdf")
        self.assertEqual(result[0].page, 37)
        self.assertEqual(result[0].epa_url, "https://epa.example/record")
        self.assertGreater(result[0].score, 0)

    def test_mismatched_chunk_cache_fails_closed(self):
        index = BM25Index.build(["one", "two"])
        with self.assertRaisesRegex(ValueError, "different document counts"):
            bm25_retrieve(index, [FakeDocument("one")], "one", top_k=1, chunk_type=FakeChunk)


class RRFTests(unittest.TestCase):
    def test_shared_chunk_wins_and_is_deduplicated(self):
        shared = chunk("shared.pdf", 1, "same chunk")
        dense_only = chunk("dense.pdf", 2)
        lexical_only = chunk("lexical.pdf", 3)
        fused = reciprocal_rank_fusion(
            [[shared, dense_only], [lexical_only, shared]],
            top_k=3,
            rrf_k=60,
        )
        self.assertEqual([item.pdf_filename for item in fused], ["shared.pdf", "lexical.pdf", "dense.pdf"])
        self.assertEqual(sum(item.pdf_filename == "shared.pdf" for item in fused), 1)
        self.assertAlmostEqual(fused[0].score, 1 / 61 + 1 / 62)

    def test_duplicate_inside_one_list_counts_once(self):
        repeated = chunk("same.pdf", 9, "same text")
        fused = reciprocal_rank_fusion([[repeated, repeated]], top_k=2, rrf_k=10)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 1 / 11)

    def test_equal_rrf_tie_prefers_first_input_list(self):
        dense = chunk("dense.pdf", 1)
        lexical = chunk("lexical.pdf", 1)
        fused = reciprocal_rank_fusion([[dense], [lexical]], top_k=2, rrf_k=60)
        self.assertEqual([item.pdf_filename for item in fused], ["dense.pdf", "lexical.pdf"])

    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([[], []], top_k=3), [])


class ConfigurationTests(unittest.TestCase):
    def test_bm25_cache_config_isolates_chunk_and_parameters(self):
        base = bm25_cache_config("chunks-a", BM25Config())
        self.assertNotEqual(base, bm25_cache_config("chunks-b", BM25Config()))
        self.assertNotEqual(base, bm25_cache_config("chunks-a", BM25Config(k1=1.5)))
        self.assertEqual(base["tokenizer_policy"], TOKENIZER_UNICODE_ALNUM_V1)

    def test_retrieval_config_records_every_ranking_setting(self):
        value = retrieval_config(
            "hybrid",
            candidate_k=30,
            top_k=6,
            rrf_k=60,
            bm25_config=BM25Config(k1=1.2, b=0.75),
        )
        self.assertEqual(value["mode"], "hybrid")
        self.assertEqual(value["candidate_k"], 30)
        self.assertEqual(value["top_k"], 6)
        self.assertEqual(value["fusion"], "reciprocal-rank-fusion")
        self.assertEqual(value["rrf_k"], 60)
        self.assertEqual(value["bm25"]["k1"], 1.2)
        self.assertEqual(value["bm25"]["b"], 0.75)

    def test_invalid_configuration_fails_early(self):
        with self.assertRaises(ValueError):
            BM25Config(k1=0)
        with self.assertRaises(ValueError):
            BM25Config(b=1.1)
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            retrieval_config(
                "hybrid", candidate_k=5, top_k=6, rrf_k=60, bm25_config=BM25Config()
            )


if __name__ == "__main__":
    unittest.main()

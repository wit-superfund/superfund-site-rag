import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citation_eval import extract_citations, parse_citations, verify_text
from reevaluate_csv import reevaluate_csv


@dataclass
class Chunk:
    pdf_filename: str
    page: int
    text: str = ""


CHUNKS = [
    Chunk("one.pdf", 10),
    Chunk("two.pdf", 20),
    Chunk("three.pdf", 30),
    Chunk("four.pdf", 40),
    Chunk("five.pdf", 50),
]


class ParserTests(unittest.TestCase):
    def test_grouped_source_forms(self):
        cases = {
            "[Source 3, Source 4]": [3, 4],
            "[Sources 3 and 4]": [3, 4],
            "[Source 3; 4]": [3, 4],
            "[Sources 3-5]": [3, 4, 5],
            "[Sources 1, 3-5, and 2]": [1, 3, 4, 5, 2],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = parse_citations(text)
                self.assertEqual([c.source_number for c in parsed.citations], expected)
                self.assertEqual(parsed.issues, ())

    def test_single_source_page(self):
        citation = parse_citations("Claim [Source 3, p. 30].").citations[0]
        self.assertEqual((citation.source_number, citation.page), (3, 30))

    def test_file_page(self):
        citation = parse_citations("Claim [three.pdf, p.30].").citations[0]
        self.assertEqual((citation.filename, citation.page), ("three.pdf", 30))

    def test_ambiguous_page_is_reported(self):
        parsed = parse_citations("Claim [Sources 3 and 4, p.30].")
        self.assertEqual(len(parsed.citations), 2)
        self.assertIn("ambiguous_page_qualifier", parsed.issues)

    def test_compatibility_tuple_api(self):
        self.assertEqual(
            extract_citations("[Source 3, Source 4]"),
            [("source", 2, None), ("source", 3, None)],
        )

    def test_repeated_references_remain_separate_occurrences(self):
        parsed = parse_citations("First [Source 1]. Second [Source 1].")
        self.assertEqual([c.source_number for c in parsed.citations], [1, 1])


class VerificationTests(unittest.TestCase):
    def test_lenient_uncited_matches_historical_validity(self):
        result = verify_text("A substantive uncited answer.", CHUNKS, mode="lenient")
        self.assertTrue(result.is_valid)
        self.assertFalse(result.citation_verified)
        self.assertEqual(result.status, "uncited")

    def test_strict_uncited_fails_closed(self):
        result = verify_text("A substantive uncited answer.", CHUNKS, mode="strict")
        self.assertFalse(result.is_valid)
        self.assertFalse(result.citation_verified)
        self.assertIn("missing_citation", result.reason_codes)

    def test_exact_abstention_is_valid_but_not_citation_verified(self):
        result = verify_text("NOT PRESENT", CHUNKS, mode="strict")
        self.assertTrue(result.is_valid)
        self.assertTrue(result.is_abstention)
        self.assertIsNone(result.citation_verified)
        self.assertEqual(result.status, "abstained")

    def test_mixed_abstention_fails_strict(self):
        result = verify_text("NOT PRESENT, but perhaps something happened.", CHUNKS, mode="strict")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, "mixed_abstention")

    def test_grouped_citations_are_all_checked(self):
        result = verify_text("Claim [Source 3, Source 4].", CHUNKS, mode="strict")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.citation_count, 2)
        self.assertEqual(len(result.verified_sources), 2)

    def test_out_of_range_source_fails(self):
        result = verify_text("Claim [Source 99].", CHUNKS, mode="strict")
        self.assertFalse(result.is_valid)
        self.assertIn("source_out_of_range", result.reason_codes)

    def test_source_page_mismatch_fails(self):
        result = verify_text("Claim [Source 3, p.31].", CHUNKS, mode="strict")
        self.assertFalse(result.is_valid)
        self.assertIn("source_page_mismatch", result.reason_codes)

    def test_filename_page_must_be_retrieved(self):
        good = verify_text("Claim [three.pdf, p.30].", CHUNKS, mode="strict")
        bad = verify_text("Claim [three.pdf, p.31].", CHUNKS, mode="strict")
        self.assertTrue(good.is_valid)
        self.assertFalse(bad.is_valid)


class PostHocTests(unittest.TestCase):
    def test_saved_csv_gets_non_destructive_eval_columns(self):
        rows = [
            {
                "answer": "Claim [Source 1, Source 2].",
                "citation_verified": "True",
                "num_citations": "0",
                "retrieved_sources": "one.pdf:p10 | two.pdf:p20",
            },
            {
                "answer": "NOT PRESENT",
                "citation_verified": "True",
                "num_citations": "0",
                "retrieved_sources": "one.pdf:p10",
            },
            {
                "answer": "Uncited claim.",
                "citation_verified": "True",
                "num_citations": "0",
                "retrieved_sources": "one.pdf:p10",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source.csv")
            output = Path(directory, "evaluated.csv")
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            count = reevaluate_csv(str(source), str(output), mode="strict")
            with output.open(newline="", encoding="utf-8") as handle:
                got = list(csv.DictReader(handle))

        self.assertEqual(count, 3)
        self.assertEqual(got[0]["citation_verified"], "True")
        self.assertEqual(got[0]["eval_citation_verified"], "true")
        self.assertEqual(got[0]["eval_citations_parsed"], "2")
        self.assertEqual(got[1]["eval_citation_verified"], "not_applicable")
        self.assertEqual(got[1]["eval_status"], "abstained")
        self.assertEqual(got[2]["eval_answer_valid"], "false")
        self.assertEqual(got[2]["eval_status"], "uncited")


class UpstreamBugReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the copied Unity verifier with lightweight dependency stubs."""
        package = types.ModuleType("snapshot_rag")
        package.__path__ = []
        sys.modules["snapshot_rag"] = package

        retrieve = types.ModuleType("snapshot_rag.retrieve")
        retrieve.RetrievedChunk = Chunk
        sys.modules["snapshot_rag.retrieve"] = retrieve

        generate = types.ModuleType("snapshot_rag.generate")

        @dataclass
        class Answer:
            text: str

        generate.Answer = Answer
        sys.modules["snapshot_rag.generate"] = generate
        cls.Answer = Answer

        snapshot = ROOT / "rag_march6" / "verify.py"
        spec = importlib.util.spec_from_file_location("snapshot_rag.verify", snapshot)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.upstream = module

    def test_actual_upstream_misses_grouped_citations(self):
        self.assertEqual(self.upstream.extract_citations("Claim [Source 3, Source 4]."), [])

    def test_actual_upstream_marks_uncited_substantive_answer_valid(self):
        result = self.upstream.verify_answer(self.Answer("Uncited claim."), CHUNKS)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.verified_sources, [])
        self.assertEqual(result.failed_citations, [])


if __name__ == "__main__":
    unittest.main()

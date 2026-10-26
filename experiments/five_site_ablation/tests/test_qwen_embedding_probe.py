from __future__ import annotations

import pathlib
import sys
import unittest


EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from qwen_embedding_probe import vector_comparison, vector_sha256  # noqa: E402


class VectorComparisonTests(unittest.TestCase):
    def test_identical_vectors_are_detected(self):
        result = vector_comparison([1.0, 0.0], [1.0, 0.0])
        self.assertTrue(result["exact_equal"])
        self.assertEqual(result["cosine_similarity"], 1.0)
        self.assertEqual(result["max_abs_delta"], 0.0)

    def test_distinct_vectors_have_distinct_hashes_and_delta(self):
        raw = [1.0, 0.0]
        instructed = [0.0, 1.0]
        result = vector_comparison(raw, instructed)
        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["cosine_similarity"], 0.0)
        self.assertEqual(result["max_abs_delta"], 1.0)
        self.assertNotEqual(vector_sha256(raw), vector_sha256(instructed))

    def test_rejects_empty_or_mismatched_vectors(self):
        with self.assertRaises(ValueError):
            vector_comparison([], [])
        with self.assertRaises(ValueError):
            vector_comparison([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

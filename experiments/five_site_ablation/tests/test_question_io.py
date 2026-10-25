from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


INTEGRATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_DIR))

from question_io import load_questions  # noqa: E402


class QuestionFileTests(unittest.TestCase):
    def test_stable_ids_survive_reordering(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "questions.tsv"
            path.write_text("health\tHealth question?\nhistory\tHistory question?\n", encoding="utf-8")
            first = load_questions(path)
            path.write_text("history\tHistory question?\nhealth\tHealth question?\n", encoding="utf-8")
            second = load_questions(path)
        self.assertEqual(dict(first), dict(second))
        self.assertNotEqual(first, second)

    def test_legacy_plain_lines_get_explicit_positional_ids(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "questions.txt"
            path.write_text("# note\nFirst?\n\nSecond?\n", encoding="utf-8")
            self.assertEqual(load_questions(path), [("q0", "First?"), ("q1", "Second?")])

    def test_duplicate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "questions.tsv"
            path.write_text("same\tFirst?\nsame\tSecond?\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate question_id"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()

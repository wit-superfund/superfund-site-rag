from __future__ import annotations

import pathlib
import sys
import unittest


EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from pdf_extract import extract_pdf_pages  # noqa: E402


class FakeMuPage:
    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text


class FakeMuDoc:
    def __init__(self, texts):
        self.pages = [FakeMuPage(text) for text in texts]
        self.page_count = len(self.pages)
        self.closed = False

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


class FakeMuModule:
    def __init__(self, texts):
        self.texts = texts
        self.doc = None

    def open(self, _path):
        self.doc = FakeMuDoc(self.texts)
        return self.doc


class FakePyPage:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error

    def extract_text(self):
        if self.error:
            raise self.error
        return self.text


class FakeReader:
    def __init__(self, pages):
        self.pages = pages


class PdfExtractionTests(unittest.TestCase):
    def test_whole_document_failure_uses_pymupdf(self):
        def broken_reader(_path):
            raise RuntimeError("AES support unavailable")

        fallback = FakeMuModule(["first", "second", "third"])
        result = extract_pdf_pages(
            "encrypted.pdf",
            max_pages=2,
            pdf_reader_cls=broken_reader,
            pymupdf_module=fallback,
        )
        self.assertEqual(result.pages, [(1, "first"), (2, "second")])
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.whole_document_fallback)
        self.assertEqual(result.fallback_page_count, 2)
        self.assertEqual(result.failed_page_count, 0)
        self.assertFalse(result.failed_document)
        self.assertTrue(fallback.doc.closed)

    def test_one_page_failure_preserves_page_number(self):
        reader = FakeReader([
            FakePyPage("primary first"),
            FakePyPage(error=ValueError("bad page")),
        ])
        result = extract_pdf_pages(
            "mixed.pdf",
            pdf_reader_cls=lambda _path: reader,
            pymupdf_module=FakeMuModule(["fallback first", "fallback second"]),
        )
        self.assertEqual(result.pages, [(1, "primary first"), (2, "fallback second")])
        self.assertEqual(result.fallback_page_count, 1)
        self.assertEqual(result.failed_page_count, 0)

    def test_both_parsers_failing_is_explicit(self):
        class BrokenMuModule:
            @staticmethod
            def open(_path):
                raise OSError("unreadable")

        def broken_reader(_path):
            raise ValueError("bad pdf")

        result = extract_pdf_pages(
            "broken.pdf",
            pdf_reader_cls=broken_reader,
            pymupdf_module=BrokenMuModule(),
        )
        self.assertEqual(result.pages, [])
        self.assertTrue(result.failed_document)
        self.assertIn("bad pdf", result.primary_error)
        self.assertIn("unreadable", result.fallback_error)


if __name__ == "__main__":
    unittest.main()

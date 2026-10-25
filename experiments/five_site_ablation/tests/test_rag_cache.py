import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from rag_cache import (
    atomic_write_csv,
    config_key,
    fingerprint_source,
    get_or_build_directory,
    get_or_build_json_gzip,
)


class RagCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.temp.cleanup()

    def test_config_key_is_stable_and_sensitive(self):
        self.assertEqual(config_key({"a": 1, "b": 2}), config_key({"b": 2, "a": 1}))
        self.assertNotEqual(config_key({"a": 1}), config_key({"a": 2}))

    def test_source_fingerprint_hashes_content_then_uses_memo(self):
        source = self.root / "source"
        source.mkdir()
        target = source / "doc.txt"
        target.write_text("first", encoding="utf-8")

        first_digest, first = fingerprint_source(source, ".txt", self.cache)
        second_digest, second = fingerprint_source(source, ".txt", self.cache)
        self.assertFalse(first["hit"])
        self.assertTrue(second["hit"])
        self.assertEqual(first_digest, second_digest)

        target.write_text("second version", encoding="utf-8")
        os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000))
        third_digest, third = fingerprint_source(source, ".txt", self.cache)
        self.assertFalse(third["hit"])
        self.assertNotEqual(first_digest, third_digest)

    def test_source_fingerprint_suffix_is_case_insensitive(self):
        source = self.root / "source"
        source.mkdir()
        (source / "upper.PDF").write_bytes(b"pdf")
        _, details = fingerprint_source(source, ".pdf", self.cache)
        self.assertEqual(details["num_files"], 1)

    def test_json_cache_hits_and_config_change_misses(self):
        calls = []

        def builder():
            calls.append("built")
            return {"value": len(calls)}

        value1, event1 = get_or_build_json_gzip(self.cache, "chunks", {"size": 10}, builder)
        value2, event2 = get_or_build_json_gzip(self.cache, "chunks", {"size": 10}, builder)
        value3, event3 = get_or_build_json_gzip(self.cache, "chunks", {"size": 11}, builder)
        self.assertEqual(value1, value2)
        self.assertFalse(event1.hit)
        self.assertTrue(event2.hit)
        self.assertFalse(event3.hit)
        self.assertEqual(value3["value"], 2)
        self.assertEqual(calls, ["built", "built"])

    def test_failed_build_never_becomes_a_cache_hit(self):
        def failed_builder():
            raise RuntimeError("simulated preemption")

        with self.assertRaisesRegex(RuntimeError, "simulated preemption"):
            get_or_build_json_gzip(self.cache, "extraction", {"site": "x"}, failed_builder)

        value, event = get_or_build_json_gzip(
            self.cache, "extraction", {"site": "x"}, lambda: {"complete": True})
        self.assertEqual(value, {"complete": True})
        self.assertFalse(event.hit)

    def test_corrupt_completed_artifact_is_quarantined_and_rebuilt(self):
        config = {"site": "x", "max_pages": 50}
        _, first = get_or_build_json_gzip(self.cache, "extraction", config, lambda: ["old"])
        Path(first.path, "READY").write_text("wrong-key\n", encoding="utf-8")

        value, second = get_or_build_json_gzip(self.cache, "extraction", config, lambda: ["new"])
        self.assertEqual(value, ["new"])
        self.assertFalse(second.hit)
        quarantined = list(Path(first.path).parent.glob(f"{Path(first.path).name}.invalid-*"))
        self.assertEqual(len(quarantined), 1)

    def test_directory_payload_is_atomic_and_reusable(self):
        calls = []

        def builder(path):
            calls.append("built")
            (path / "index.bin").write_bytes(b"index-data")
            return "in-memory-index", {"num_vectors": 7}

        def loader(path):
            return (path / "index.bin").read_bytes()

        value1, event1 = get_or_build_directory(self.cache, "faiss", {"model": "m1"}, loader, builder)
        value2, event2 = get_or_build_directory(self.cache, "faiss", {"model": "m1"}, loader, builder)
        self.assertEqual(value1, "in-memory-index")
        self.assertEqual(value2, b"index-data")
        self.assertFalse(event1.hit)
        self.assertTrue(event2.hit)
        self.assertEqual(calls, ["built"])

    def test_directory_loader_failure_does_not_quarantine_valid_cache(self):
        def builder(path):
            (path / "index.bin").write_bytes(b"index-data")
            return "built", None

        _, event = get_or_build_directory(
            self.cache, "faiss", {"model": "m1"}, lambda path: b"loaded", builder)

        def runtime_failure(path):
            raise RuntimeError("simulated GPU OOM")

        with self.assertRaisesRegex(RuntimeError, "GPU OOM"):
            get_or_build_directory(self.cache, "faiss", {"model": "m1"}, runtime_failure, builder)
        self.assertTrue(Path(event.path).is_dir())
        self.assertEqual(list(Path(event.path).parent.glob(f"{Path(event.path).name}.invalid-*")), [])

    def test_atomic_csv_replaces_complete_file(self):
        output = self.root / "answers.csv"
        output.write_text("stale,partial\n", encoding="utf-8")
        atomic_write_csv(output, ["id", "answer"], [{"id": 1, "answer": "yes"}])
        with open(output, newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [{"id": "1", "answer": "yes"}])
        self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

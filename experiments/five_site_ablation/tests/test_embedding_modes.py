import asyncio
import json
import pathlib
import sys
import tempfile
import unittest


EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from embedding_modes import (  # noqa: E402
    EmbeddingProfile,
    QueryAwareEmbeddings,
    SUPERFUND_QUERY_INSTRUCTION,
    build_embedding_profile,
    qwen3_query_text,
    validate_embedding_model_assets,
)


class SpyEmbeddings:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [float(len(text))]


class AsyncSpyEmbeddings(SpyEmbeddings):
    async def aembed_documents(self, texts):
        self.document_calls.append(texts)
        return [[2.0] for _ in texts]

    async def aembed_query(self, text):
        self.query_calls.append(text)
        return [3.0]


class EmbeddingProfileTests(unittest.TestCase):
    def test_raw_profile_preserves_legacy_query_and_index_key(self):
        profile = build_embedding_profile("qwen3-embed-8b", "none")
        self.assertEqual(profile.transform_query("Where is the tailings pond?"), "Where is the tailings pond?")
        self.assertEqual(profile.index_key, "qwen3-embed-8b")
        self.assertEqual(profile.manifest_fields()["query_instruction"], "")

    def test_superfund_profile_uses_official_qwen_format(self):
        profile = build_embedding_profile("qwen3-embed-8b", "superfund")
        self.assertEqual(
            profile.transform_query("Which contaminants were detected?"),
            "Instruct: "
            + SUPERFUND_QUERY_INSTRUCTION
            + "\nQuery:Which contaminants were detected?",
        )

    def test_documents_are_never_instructed(self):
        spy = SpyEmbeddings()
        adapter = QueryAwareEmbeddings(spy, build_embedding_profile("qwen3-embed-8b", "superfund"))
        documents = ["Mine waste entered the river.", "Arsenic was detected."]
        adapter.embed_documents(documents)
        adapter.embed_query("What entered the river?")
        self.assertEqual(spy.document_calls, [documents])
        self.assertTrue(spy.query_calls[0].startswith("Instruct: "))

    def test_callable_compatibility_path_uses_query_transform(self):
        spy = SpyEmbeddings()
        profile = build_embedding_profile("qwen3-embed-8b", "superfund")
        adapter = QueryAwareEmbeddings(spy, profile)
        transformed = profile.transform_query("When did mining begin?")
        self.assertEqual(adapter("When did mining begin?"), [float(len(transformed))])
        self.assertEqual(spy.query_calls, [profile.transform_query("When did mining begin?")])

    def test_instructed_index_key_is_deterministic_and_cannot_collide_with_raw(self):
        raw = EmbeddingProfile("qwen3-embed-8b", "none")
        first = EmbeddingProfile("qwen3-embed-8b", "superfund")
        second = EmbeddingProfile("qwen3-embed-8b", "superfund")
        custom = EmbeddingProfile("qwen3-embed-8b", "superfund", "Retrieve passages about remediation.")
        self.assertEqual(first.index_key, second.index_key)
        self.assertEqual(first.index_key, "qwen3-embed-8b__query-superfund-fc6e7e118938")
        self.assertNotEqual(first.index_key, raw.index_key)
        self.assertNotEqual(first.index_key, custom.index_key)

    def test_instruction_hash_is_recorded(self):
        profile = EmbeddingProfile("qwen3-embed-8b", "superfund")
        fields = profile.manifest_fields()
        self.assertEqual(fields["query_instruction_mode"], "superfund")
        self.assertEqual(fields["query_instruction"], SUPERFUND_QUERY_INSTRUCTION)
        self.assertEqual(len(fields["query_instruction_sha256"]), 64)
        self.assertEqual(fields["embedding_index_key"], profile.index_key)

    def test_instruction_mode_rejects_non_qwen_model(self):
        with self.assertRaisesRegex(ValueError, "only for qwen3"):
            EmbeddingProfile("e5-mistral-7b", "superfund")

    def test_none_mode_rejects_hidden_custom_instruction(self):
        with self.assertRaisesRegex(ValueError, "must be empty"):
            EmbeddingProfile("qwen3-embed-8b", "none", "unexpected")

    def test_empty_qwen_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            qwen3_query_text("question", "   ")

    def test_async_adapter_uses_same_query_transform(self):
        spy = AsyncSpyEmbeddings()
        adapter = QueryAwareEmbeddings(spy, EmbeddingProfile("qwen3-embed-8b", "superfund"))
        asyncio.run(adapter.aembed_query("What was remediated?"))
        self.assertEqual(
            spy.query_calls,
            ["Instruct: " + SUPERFUND_QUERY_INSTRUCTION + "\nQuery:What was remediated?"],
        )

    def test_incomplete_qwen_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            pathlib.Path(directory, "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_embedding_model_assets("qwen3-embed-8b", directory)

    def test_complete_qwen_embedding_interface_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            values = {
                "config.json": {"model_type": "qwen3", "hidden_size": 4096},
                "config_sentence_transformers.json": {"similarity_fn_name": "cosine"},
                "modules.json": [
                    {"type": "sentence_transformers.models.Transformer"},
                    {"type": "sentence_transformers.models.Pooling"},
                    {"type": "sentence_transformers.models.Normalize"},
                ],
                "tokenizer.json": {"model": {"type": "BPE"}},
                "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
                "1_Pooling/config.json": {"pooling_mode_lasttoken": True},
            }
            for name, value in values.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            manifest = validate_embedding_model_assets("qwen3-embed-8b", directory)
            self.assertEqual(manifest["pooling_mode"], "lasttoken")
            self.assertEqual(len(manifest["model_asset_bundle_sha256"]), 64)


class StagedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.compare_text = (EXPERIMENT_DIR / "compare_rag.py").read_text(encoding="utf-8")
        self.runner_text = (EXPERIMENT_DIR / "run_ablation_arm.slurm").read_text(encoding="utf-8")
        self.matrix_text = (EXPERIMENT_DIR / "ablation_matrix.tsv").read_text(encoding="utf-8")

    def test_compare_cli_records_mode_and_uses_profile_index_key(self):
        self.assertIn('"--query_instruction_mode"', self.compare_text)
        self.assertIn('"query_instruction_mode": embedding_profile.query_mode', self.compare_text)
        self.assertIn('"embedding_model_path": (', self.compare_text)
        self.assertIn('embedding_profile.index_key', self.compare_text)

    def test_raw_mode_keeps_original_embedding_object(self):
        self.assertIn(
            "base_embeddings\n            if embedding_profile.query_mode == QUERY_MODE_NONE",
            self.compare_text,
        )

    def test_runner_forwards_frozen_query_configuration(self):
        self.assertIn('--query_instruction_mode "$QUERY_MODE"', self.runner_text)
        self.assertIn("query_instruction_sha256", self.runner_text)
        self.assertIn("fc6e7e118938eafb71009851781003cbab03129eee13cf942616ec1265d3a340", self.runner_text)
        self.assertIn("I_qwen\tindividual\tready\t50\tqwen3-embed-8b\tsuperfund", self.matrix_text)


if __name__ == "__main__":
    unittest.main()

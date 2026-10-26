"""Embedding profiles for reproducible dense-retrieval ablations.

The underlying embedding model still receives documents unchanged.  A profile
may transform *queries only*, which is how Qwen3-Embedding expects retrieval
instructions to be supplied.

This module deliberately depends only on the Python standard library.  The
``QueryAwareEmbeddings`` adapter is compatible with LangChain's embeddings
protocol through duck typing, while the lightweight implementation remains
unit-testable without loading LangChain, PyTorch, or a model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    from langchain_core.embeddings import Embeddings as _LangChainEmbeddings
except ImportError:  # Keep pure configuration/unit tests dependency-free.
    class _LangChainEmbeddings:  # type: ignore[no-redef]
        pass


QUERY_MODE_NONE = "none"
QUERY_MODE_SUPERFUND = "superfund"
QUERY_MODES = (QUERY_MODE_NONE, QUERY_MODE_SUPERFUND)
QWEN3_ASSET_SCHEMA = "qwen3-embedding-assets-v1"
QWEN3_REQUIRED_ASSETS = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
)

# Qwen recommends one English sentence describing the retrieval task.  Keep
# this text versioned through its hash in manifests and instructed index paths.
SUPERFUND_QUERY_INSTRUCTION = (
    "Given a question about a U.S. Superfund site, retrieve passages from its "
    "environmental and historical records that contain evidence needed to answer the question."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_embedding_model_assets(model_key: str, model_path: str) -> Dict[str, Any]:
    """Fail closed when a Qwen checkpoint lacks its embedding interface.

    A raw causal-LM snapshot can be opened by Sentence Transformers, but its
    automatically constructed tokenizer/pooler may silently return unusable
    embeddings. Qwen3 therefore requires the official tokenizer, module graph,
    last-token pooler, and normalization metadata before a benchmark can run.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"embedding model path is not a directory: {root}")
    if model_key != "qwen3-embed-8b":
        return {
            "model_asset_schema": "directory-present-v1",
            "model_asset_path": str(root.resolve()),
        }

    missing = [name for name in QWEN3_REQUIRED_ASSETS if not (root / name).is_file()]
    if missing:
        raise ValueError(
            "Qwen3 embedding checkpoint is incomplete; missing official embedding assets: "
            + ", ".join(missing)
        )
    with (root / "config.json").open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    with (root / "modules.json").open(encoding="utf-8") as handle:
        modules = json.load(handle)
    with (root / "1_Pooling/config.json").open(encoding="utf-8") as handle:
        pooling = json.load(handle)
    with (root / "tokenizer.json").open(encoding="utf-8") as handle:
        tokenizer = json.load(handle)

    module_types = {item.get("type") for item in modules}
    expected_types = {
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    }
    if not expected_types.issubset(module_types):
        raise ValueError("Qwen3 modules.json is missing Transformer, Pooling, or Normalize")
    if pooling.get("pooling_mode_lasttoken") is not True:
        raise ValueError("Qwen3 embedding pooler must use last-token pooling")
    if any(
        pooling.get(name) is True
        for name in (
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_weightedmean_tokens",
        )
    ):
        raise ValueError("Qwen3 embedding pooler enables an unexpected additional pooling mode")
    if model_config.get("model_type") != "qwen3" or model_config.get("hidden_size") != 4096:
        raise ValueError("unexpected Qwen3-Embedding-8B model configuration")
    if not isinstance(tokenizer.get("model"), dict):
        raise ValueError("Qwen3 tokenizer.json has no tokenizer model")

    asset_hashes = {
        name: _file_sha256(root / name)
        for name in QWEN3_REQUIRED_ASSETS
    }
    bundle = hashlib.sha256(
        json.dumps(asset_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "model_asset_schema": QWEN3_ASSET_SCHEMA,
        "model_asset_path": str(root.resolve()),
        "model_asset_bundle_sha256": bundle,
        "model_asset_hashes": asset_hashes,
        "pooling_mode": "lasttoken",
        "normalization": True,
    }


def qwen3_query_text(query: str, instruction: str) -> str:
    """Return the exact query-side format recommended by Qwen3-Embedding."""
    if not instruction or not instruction.strip():
        raise ValueError("Qwen3 query instruction must be non-empty")
    return f"Instruct: {instruction.strip()}\nQuery:{query}"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "custom"


@dataclass(frozen=True)
class EmbeddingProfile:
    """Configuration that controls dense document and query encoding."""

    model_key: str
    query_mode: str = QUERY_MODE_NONE
    query_instruction: Optional[str] = None

    def __post_init__(self) -> None:
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"Unknown query mode: {self.query_mode!r}")
        if self.query_mode == QUERY_MODE_NONE:
            if self.query_instruction not in (None, ""):
                raise ValueError("query_instruction must be empty when query_mode='none'")
            return
        if self.model_key != "qwen3-embed-8b":
            raise ValueError("The instructed query mode is currently supported only for qwen3-embed-8b")
        if not self.instruction:
            raise ValueError("The instructed query mode requires a non-empty instruction")

    @property
    def instruction(self) -> Optional[str]:
        if self.query_mode == QUERY_MODE_NONE:
            return None
        return self.query_instruction or SUPERFUND_QUERY_INSTRUCTION

    @property
    def instruction_sha256(self) -> str:
        return hashlib.sha256((self.instruction or "").encode("utf-8")).hexdigest()

    @property
    def index_key(self) -> str:
        """Stable index component; legacy/raw mode preserves its old name."""
        if self.query_mode == QUERY_MODE_NONE:
            return self.model_key
        assert self.instruction is not None
        return f"{self.model_key}__query-{_slug(self.query_mode)}-{_short_hash(self.instruction)}"

    def transform_query(self, query: str) -> str:
        if self.query_mode == QUERY_MODE_NONE:
            return query
        assert self.instruction is not None
        return qwen3_query_text(query, self.instruction)

    def manifest_fields(self) -> Mapping[str, str]:
        return {
            "query_instruction_mode": self.query_mode,
            "query_instruction": self.instruction or "",
            "query_instruction_sha256": self.instruction_sha256,
            "embedding_index_key": self.index_key,
        }


class QueryAwareEmbeddings(_LangChainEmbeddings):
    """Wrap a dense embedder while transforming queries independently.

    ``embed_documents`` is a direct pass-through.  ``embed_query`` applies the
    profile first.  This preserves the query/document asymmetry required by
    Qwen3 and leaves the retriever interface unchanged for future hybrid/RRF
    composition.
    """

    def __init__(self, delegate: Any, profile: EmbeddingProfile):
        self.delegate = delegate
        self.profile = profile

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.delegate.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.delegate.embed_query(self.profile.transform_query(text))

    def __call__(self, text: str) -> List[float]:
        """Support LangChain FAISS versions that treat adapters as callables.

        Some ``langchain-community`` releases recognize only concrete
        ``Embeddings`` subclasses at query time.  They otherwise invoke the
        stored embedding object directly, so route that compatibility path
        through the same query-only transformation.
        """
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        method = getattr(self.delegate, "aembed_documents", None)
        if method is not None:
            return await method(texts)
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        transformed = self.profile.transform_query(text)
        method = getattr(self.delegate, "aembed_query", None)
        if method is not None:
            return await method(transformed)
        return self.delegate.embed_query(transformed)


def build_embedding_profile(
    model_key: str,
    query_mode: str,
    query_instruction: Optional[str] = None,
) -> EmbeddingProfile:
    """Construct a validated profile from CLI/config values."""
    return EmbeddingProfile(
        model_key=model_key,
        query_mode=query_mode,
        query_instruction=query_instruction,
    )


def manifest_dict(profile: EmbeddingProfile) -> Dict[str, str]:
    """Materialize profile metadata for JSON/CSV writers."""
    return dict(profile.manifest_fields())

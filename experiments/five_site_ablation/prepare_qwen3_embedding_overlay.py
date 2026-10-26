#!/usr/bin/env python3
"""Build a complete Qwen3 embedding view over Unity's weight-only snapshot.

The shared checkpoint has the 8B weights but omits the Sentence Transformers
module graph, last-token pooler, and tokenizer vocabulary. Download those six
small files from the same frozen Hugging Face revision, place them in an asset
directory, then run this script. Large weight files are symlinked, not copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
ASSET_HASHES = {
    "config_sentence_transformers.json": "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "pooling_config.json": "2e1da26b3fd65cf7e370d2fabf28a8c59efa7edb525d1b8b50be8e5ca1048ea6",
    "tokenizer.json": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
LINK_NAMES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checked_asset(asset_dir: Path, name: str) -> Path:
    path = asset_dir / name
    if not path.is_file():
        raise ValueError(f"missing downloaded asset: {path}")
    actual = sha256_file(path)
    if actual != ASSET_HASHES[name]:
        raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
    return path


def ensure_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source.resolve())


def copy_verified(source: Path, destination: Path) -> None:
    if destination.is_file():
        if sha256_file(destination) == sha256_file(source):
            return
        raise ValueError(f"refusing to replace different file: {destination}")
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def build_overlay(source: Path, destination: Path, asset_dir: Path) -> dict:
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not (source / "config.json").is_file():
        raise ValueError(f"source checkpoint is missing config.json: {source}")
    model_parts = sorted(source.glob("model-*-of-*.safetensors"))
    if not model_parts:
        raise ValueError(f"source checkpoint has no sharded model weights: {source}")
    for name in LINK_NAMES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"source checkpoint is missing {name}: {source}")
        ensure_symlink(path, destination / name)
    for path in model_parts:
        ensure_symlink(path, destination / path.name)

    for name in ASSET_HASHES:
        target = destination / name
        if name == "pooling_config.json":
            target = destination / "1_Pooling" / "config.json"
            target.parent.mkdir(exist_ok=True)
        copy_verified(checked_asset(asset_dir, name), target)
    (destination / "2_Normalize").mkdir(exist_ok=True)

    provenance = {
        "schema": "qwen3-embedding-overlay-v1",
        "model": "Qwen/Qwen3-Embedding-8B",
        "revision": REVISION,
        "weight_source": str(source),
        "asset_sha256": ASSET_HASHES,
        "weight_parts": [path.name for path in model_parts],
    }
    provenance_path = destination / "MODEL_PROVENANCE.json"
    encoded = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if provenance_path.exists() and provenance_path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"refusing to replace different provenance: {provenance_path}")
    provenance_path.write_text(encoded, encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_overlay(args.source, args.destination, args.assets), sort_keys=True))


if __name__ == "__main__":
    main()

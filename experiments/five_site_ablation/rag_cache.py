"""Content-addressed, preemption-safe cache primitives for the RAG benchmark.

The cache stores immutable artifact directories.  A directory is visible to
readers only after all payload files and a READY marker have been written in a
temporary sibling and atomically renamed into place.  Per-key advisory locks
prevent two Slurm jobs from publishing the same artifact concurrently; locks
are released automatically if a job is killed.

This module deliberately has no LangChain, FAISS, Torch, or PDF dependencies so
its invalidation and atomicity behavior can be tested on a login node.
"""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


CACHE_FORMAT_VERSION = 1


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for cache keys."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_key(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync (not supported by every shared filesystem)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


@contextlib.contextmanager
def cache_lock(cache_root: str | os.PathLike[str], namespace: str, key: str) -> Iterator[None]:
    lock_dir = Path(cache_root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{namespace}-{key}.lock"
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _artifact_path(cache_root: str | os.PathLike[str], namespace: str, key: str) -> Path:
    return Path(cache_root) / namespace / key[:2] / key


def _metadata(path: Path) -> dict[str, Any]:
    with open(path / "metadata.json", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_artifact(path: Path, key: str, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if (path / "READY").read_text(encoding="utf-8").strip() != key:
        raise ValueError(f"cache READY marker mismatch: {path}")
    metadata = _metadata(path)
    if metadata.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported cache format: {path}")
    if metadata.get("key") != key or metadata.get("config") != config:
        raise ValueError(f"cache configuration mismatch: {path}")
    return metadata


def _quarantine_invalid(path: Path) -> None:
    if not path.exists():
        return
    quarantined = path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex}")
    os.replace(path, quarantined)
    print(f"  quarantined invalid cache artifact: {quarantined}", flush=True)


def _publish_artifact(
    destination: Path,
    key: str,
    config: dict[str, Any],
    writer: Callable[[Path], dict[str, Any] | None],
) -> tuple[dict[str, Any], Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        details = writer(tmp) or {}
        metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "key": key,
            "config": config,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "details": details,
        }
        _atomic_write_text(tmp / "metadata.json", canonical_json(metadata) + "\n")
        _atomic_write_text(tmp / "READY", key + "\n")
        _fsync_directory(tmp)
        if destination.exists():
            _quarantine_invalid(destination)
        os.replace(tmp, destination)
        _fsync_directory(destination.parent)
        return metadata, destination
    except BaseException:
        # A hard preemption can leave this uniquely named temp directory behind,
        # but it can never be mistaken for a completed artifact.
        shutil.rmtree(tmp, ignore_errors=True)
        raise


@dataclass(frozen=True)
class CacheEvent:
    namespace: str
    key: str
    hit: bool
    elapsed_sec: float
    build_sec: float
    path: str
    metadata: dict[str, Any]


def get_or_build_json_gzip(
    cache_root: str | os.PathLike[str],
    namespace: str,
    config: dict[str, Any],
    builder: Callable[[], Any],
) -> tuple[Any, CacheEvent]:
    """Load a JSON payload or build and atomically publish it."""
    started = time.monotonic()
    key = config_key(config)
    path = _artifact_path(cache_root, namespace, key)

    def load() -> tuple[Any, dict[str, Any]]:
        metadata = _validate_artifact(path, key, config)
        with gzip.open(path / "payload.json.gz", "rt", encoding="utf-8") as handle:
            return json.load(handle), metadata

    try:
        payload, metadata = load()
        elapsed = time.monotonic() - started
        return payload, CacheEvent(namespace, key, True, elapsed, 0.0, str(path), metadata)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, EOFError):
        pass

    with cache_lock(cache_root, namespace, key):
        try:
            payload, metadata = load()
            elapsed = time.monotonic() - started
            return payload, CacheEvent(namespace, key, True, elapsed, 0.0, str(path), metadata)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, EOFError):
            if path.exists():
                _quarantine_invalid(path)

        build_started = time.monotonic()
        payload = builder()
        build_sec = time.monotonic() - build_started

        def write(tmp: Path) -> dict[str, Any]:
            with open(tmp / "payload.json.gz", "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
                    zipped.write(canonical_json(payload).encode("utf-8"))
                raw.flush()
                os.fsync(raw.fileno())
            return {"payload_bytes": (tmp / "payload.json.gz").stat().st_size}

        metadata, _ = _publish_artifact(path, key, config, write)
        elapsed = time.monotonic() - started
        return payload, CacheEvent(namespace, key, False, elapsed, build_sec, str(path), metadata)


def get_or_build_directory(
    cache_root: str | os.PathLike[str],
    namespace: str,
    config: dict[str, Any],
    loader: Callable[[Path], Any],
    builder: Callable[[Path], tuple[Any, dict[str, Any] | None]],
) -> tuple[Any, CacheEvent]:
    """Load or atomically publish a directory payload (for example, FAISS)."""
    started = time.monotonic()
    key = config_key(config)
    path = _artifact_path(cache_root, namespace, key)

    def validated_metadata() -> dict[str, Any] | None:
        try:
            return _validate_artifact(path, key, config)
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    metadata = validated_metadata()
    if metadata is not None:
        # Loader failures (for example CUDA OOM or incompatible runtime) are
        # not evidence of cache corruption and must not quarantine a valid index.
        value = loader(path)
        elapsed = time.monotonic() - started
        return value, CacheEvent(namespace, key, True, elapsed, 0.0, str(path), metadata)

    with cache_lock(cache_root, namespace, key):
        metadata = validated_metadata()
        if metadata is not None:
            value = loader(path)
            elapsed = time.monotonic() - started
            return value, CacheEvent(namespace, key, True, elapsed, 0.0, str(path), metadata)
        if path.exists():
            _quarantine_invalid(path)

        build_started = time.monotonic()
        built_value: Any = None
        built_details: dict[str, Any] | None = None

        def write(tmp: Path) -> dict[str, Any] | None:
            nonlocal built_value, built_details
            built_value, built_details = builder(tmp)
            return built_details

        metadata, _ = _publish_artifact(path, key, config, write)
        build_sec = time.monotonic() - build_started
        elapsed = time.monotonic() - started
        return built_value, CacheEvent(namespace, key, False, elapsed, build_sec, str(path), metadata)


def fingerprint_source(
    root: str | os.PathLike[str],
    suffix: str,
    cache_root: str | os.PathLike[str],
    force_rehash: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return a content fingerprint, memoizing full hashes behind a stat key.

    The first encounter hashes every source byte.  Later calls avoid rereading
    unchanged files when relative path, size, and nanosecond mtime all match.
    Pass ``force_rehash`` for an audit run that distrusts filesystem metadata.
    """
    started = time.monotonic()
    root_path = Path(root).resolve()
    wanted_suffix = suffix.lower()
    files = sorted(
        path for path in root_path.rglob("*")
        if path.is_file() and path.name.lower().endswith(wanted_suffix)
    )
    inventory = [
        {
            "relative_path": path.relative_to(root_path).as_posix(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    stat_config = {
        "schema": "source-fingerprint-stat-v1",
        "root": str(root_path),
        "suffix": suffix.lower(),
        "inventory": inventory,
    }
    stat_key = config_key(stat_config)
    memo = Path(cache_root) / "source_fingerprints" / stat_key[:2] / f"{stat_key}.json"

    def read_memo() -> dict[str, Any]:
        with open(memo, encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("stat_config") != stat_config:
            raise ValueError("source fingerprint memo mismatch")
        return value

    if not force_rehash:
        try:
            value = read_memo()
            return value["content_sha256"], {
                "hit": True,
                "elapsed_sec": time.monotonic() - started,
                "num_files": len(files),
                "num_bytes": sum(item["size"] for item in inventory),
                "stat_key": stat_key,
            }
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    with cache_lock(cache_root, "source-fingerprint", stat_key):
        if not force_rehash:
            try:
                value = read_memo()
                return value["content_sha256"], {
                    "hit": True,
                    "elapsed_sec": time.monotonic() - started,
                    "num_files": len(files),
                    "num_bytes": sum(item["size"] for item in inventory),
                    "stat_key": stat_key,
                }
            except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

        content_files = []
        for path, stat_item in zip(files, inventory):
            content_files.append({**stat_item, "sha256": sha256_file(path)})
        content_sha256 = config_key({
            "schema": "source-content-v1",
            "suffix": suffix.lower(),
            "files": [{"relative_path": item["relative_path"], "sha256": item["sha256"]}
                      for item in content_files],
        })
        value = {
            "stat_config": stat_config,
            "content_sha256": content_sha256,
            "files": content_files,
        }
        _atomic_write_text(memo, canonical_json(value) + "\n")
        return content_sha256, {
            "hit": False,
            "elapsed_sec": time.monotonic() - started,
            "num_files": len(files),
            "num_bytes": sum(item["size"] for item in inventory),
            "stat_key": stat_key,
        }


def atomic_write_csv(path: str | os.PathLike[str], fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a CSV without exposing a partial result to a requeued job."""
    import csv

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise

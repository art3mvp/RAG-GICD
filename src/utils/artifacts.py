from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.document_loader import TextChunk
from src.utils.hashing import stable_hash
from src.utils.io import dump_json, load_json

HYBRID_ARTIFACT_VERSION = 1
DENSE_ARTIFACT_VERSION = 1
MANIFEST_FILENAME = "hybrid_manifest.json"
DENSE_MANIFEST_FILENAME = "dense_manifest.json"
BM25_FILENAME = "lexical/bm25.pkl"


def chunks_hash(chunks: list[TextChunk]) -> str:
    payload = "\n".join(
        f"{chunk.metadata.get('chunk_id', '')}\t{chunk.text}"
        for chunk in chunks
    )
    return stable_hash(payload)


def build_manifest(chunks: list[TextChunk], settings: Any, collection_name: str) -> dict[str, Any]:
    return {
        "artifact_version": HYBRID_ARTIFACT_VERSION,
        "corpus_hash": chunks_hash(chunks),
        "chunk_count": len(chunks),
        "chunking": {
            "strategy": getattr(settings, "chunking_strategy", "fixed"),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "bm25": {"k1": settings.bm25_k1, "b": settings.bm25_b},
        "embedding_model": settings.openai_embedding_model,
        "collection_name": collection_name,
        "artifacts": {
            "chunks": str(
                Path(getattr(settings, "processed_data_dir", "data/processed")) / "chunks.jsonl"
            ),
            "bm25": BM25_FILENAME,
        },
    }


def build_dense_manifest(chunks: list[TextChunk], settings: Any, collection_name: str) -> dict[str, Any]:
    return {
        "artifact_version": DENSE_ARTIFACT_VERSION,
        "corpus_hash": chunks_hash(chunks),
        "chunk_count": len(chunks),
        "chunking": {
            "strategy": getattr(settings, "chunking_strategy", "fixed"),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "embedding_model": settings.openai_embedding_model,
        "collection_name": collection_name,
        "artifacts": {
            "chunks": str(
                Path(getattr(settings, "processed_data_dir", "data/processed")) / "chunks.jsonl"
            ),
        },
    }


def validate_dense_manifest(
    manifest: dict[str, Any],
    chunks: list[TextChunk],
    settings: Any,
    collection_name: str,
) -> None:
    if manifest.get("artifact_version") != DENSE_ARTIFACT_VERSION:
        raise RuntimeError("Unsupported dense artifact version")
    expected = build_dense_manifest(chunks, settings, collection_name)
    for key in ("corpus_hash", "chunk_count", "embedding_model", "collection_name"):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"Dense manifest mismatch for {key}")
    if manifest.get("chunking") != expected["chunking"]:
        raise RuntimeError("Dense manifest mismatch for chunking configuration")
    if manifest.get("artifacts") != expected["artifacts"]:
        raise RuntimeError("Dense manifest artifact paths are invalid")


def validate_manifest(
    manifest: dict[str, Any],
    chunks: list[TextChunk],
    settings: Any,
    collection_name: str,
) -> None:
    if manifest.get("artifact_version") != HYBRID_ARTIFACT_VERSION:
        raise RuntimeError("Unsupported hybrid artifact version")
    expected = build_manifest(chunks, settings, collection_name)
    for key in ("corpus_hash", "chunk_count", "embedding_model", "collection_name"):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"Hybrid manifest mismatch for {key}")
    if manifest.get("chunking") != expected["chunking"]:
        raise RuntimeError("Hybrid manifest mismatch for chunking configuration")
    if manifest.get("bm25") != expected["bm25"]:
        raise RuntimeError("Hybrid manifest mismatch for BM25 configuration")
    if manifest.get("artifacts") != expected["artifacts"]:
        raise RuntimeError("Hybrid manifest artifact paths are invalid")


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    dump_json(temporary, manifest)
    temporary.replace(target)


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise RuntimeError(f"Artifact manifest not found: {target}")
    return load_json(target)

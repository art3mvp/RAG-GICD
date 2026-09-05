from types import SimpleNamespace

import pytest

from src.ingestion.document_loader import TextChunk
from src.utils.artifacts import (
    build_dense_manifest,
    build_manifest,
    validate_dense_manifest,
    validate_manifest,
)


def _settings():
    return SimpleNamespace(
        chunk_size=100,
        chunk_overlap=10,
        bm25_k1=1.5,
        bm25_b=0.75,
        openai_embedding_model="test-embedding",
    )


def test_manifest_rejects_changed_processed_chunks() -> None:
    settings = _settings()
    original = [TextChunk("one", {"chunk_id": "a#0"})]
    manifest = build_manifest(original, settings, "hybrid_rag")

    with pytest.raises(RuntimeError, match="corpus_hash"):
        validate_manifest(
            manifest,
            [TextChunk("changed", {"chunk_id": "a#0"})],
            settings,
            "hybrid_rag",
        )


def test_dense_manifest_rejects_changed_embedding_model() -> None:
    settings = _settings()
    original = [TextChunk("one", {"chunk_id": "a#0"})]
    manifest = build_dense_manifest(original, settings, "naive_rag")
    settings.openai_embedding_model = "different-embedding"

    with pytest.raises(RuntimeError, match="embedding_model"):
        validate_dense_manifest(manifest, original, settings, "naive_rag")

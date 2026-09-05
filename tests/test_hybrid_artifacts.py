from types import SimpleNamespace

import pytest

from src.ingestion.document_loader import TextChunk
from src.utils.artifacts import build_manifest, validate_manifest


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

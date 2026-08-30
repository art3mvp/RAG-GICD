from __future__ import annotations

from src.ingestion.document_loader import TextChunk


def lower_for_bm25(chunks: list[TextChunk]) -> list[str]:
    return [chunk.text.lower() for chunk in chunks]

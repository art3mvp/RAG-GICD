from __future__ import annotations

import re

from src.ingestion.document_loader import TextChunk


SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def preprocess_documents(documents: list[TextChunk]) -> list[TextChunk]:
    seen: set[str] = set()
    processed: list[TextChunk] = []

    for doc in documents:
        cleaned = clean_text(doc.text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        processed.append(TextChunk(text=cleaned, metadata=dict(doc.metadata)))

    return processed

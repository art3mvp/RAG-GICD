from __future__ import annotations

import re

from src.ingestion.document_loader import TextChunk


SPACE_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    normalized_lines = [SPACE_RE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


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

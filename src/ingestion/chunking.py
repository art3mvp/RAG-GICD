from __future__ import annotations

from src.ingestion.document_loader import TextChunk


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "fixed",
) -> list[str]:
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    if strategy != "fixed":
        raise ValueError("chunking strategy must be 'fixed'")
    return _split_fixed(text, chunk_size, chunk_overlap)


def chunk_documents(
    documents: list[TextChunk],
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "fixed",
) -> list[TextChunk]:
    result: list[TextChunk] = []
    for doc in documents:
        parts = split_text(doc.text, chunk_size, chunk_overlap, strategy)
        for idx, chunk in enumerate(parts):
            metadata = dict(doc.metadata)
            source = str(metadata.get("source", "unknown"))
            page = metadata.get("page")
            location = f"page-{page}" if page is not None else "document"
            metadata["chunk_id"] = f"{source}#{location}#chunk-{idx}"
            metadata["chunk_index"] = idx
            result.append(TextChunk(text=chunk, metadata=metadata))
    return result

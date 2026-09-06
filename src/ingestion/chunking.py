from __future__ import annotations

import re

from src.ingestion.document_loader import TextChunk


STRUCTURAL_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+\S|(?:TÍTULO|TITULO|CAPÍTULO|CAPITULO|SECCIÓN|SECCION|APARTADO)\b|(?:Artículo|Articulo|Art\.)\s+\S|\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÜÑ])",
    re.IGNORECASE,
)
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)\S")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


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


def _pack_words(text: str, chunk_size: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > chunk_size:
            packed.append(current)
            current = word
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _overlap_suffix(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    words = text.split()
    suffix: list[str] = []
    length = 0
    for word in reversed(words):
        candidate_length = length + len(word) + (1 if suffix else 0)
        if suffix and candidate_length > overlap:
            break
        suffix.append(word)
        length = candidate_length
    return " ".join(reversed(suffix))


def _split_logical_unit(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        sentences = [part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip()]
        for sentence in sentences or [paragraph.strip()]:
            pieces.extend(_pack_words(sentence, chunk_size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            suffix = _overlap_suffix(current, chunk_overlap)
            overlapped = f"{suffix} {piece}".strip() if suffix else piece
            current = overlapped if len(overlapped) <= chunk_size else piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _logical_units(text: str) -> tuple[list[str], bool]:
    units: list[str] = []
    current: list[str] = []
    hierarchy: list[str] = []
    found_structure = False

    def flush() -> None:
        if current:
            units.append("\n".join(current).strip())
            current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current and STRUCTURAL_LINE_RE.match(current[-1]):
                continue
            flush()
            continue
        if STRUCTURAL_LINE_RE.match(line):
            found_structure = True
            flush()
            if re.match(r"^\s*(?:Artículo|Articulo|Art\.)\s+", line, re.IGNORECASE):
                hierarchy = [
                    item
                    for item in hierarchy
                    if not re.match(r"^(?:Artículo|Articulo|Art\.)\s+", item, re.IGNORECASE)
                ] + [stripped]
            else:
                hierarchy.append(stripped)
            current.append(stripped)
            continue
        if LIST_LINE_RE.match(line):
            found_structure = True
            if current and not LIST_LINE_RE.match(current[-1]) and not STRUCTURAL_LINE_RE.match(current[-1]):
                flush()
            current.append(stripped)
            continue
        if (
            current
            and LIST_LINE_RE.match(current[-1])
            and not STRUCTURAL_LINE_RE.match(current[-1])
        ):
            flush()
        if current:
            current.append(stripped)
        else:
            current.extend(hierarchy + [stripped] if hierarchy else [stripped])

    flush()
    return units, found_structure


def split_logical_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    """Split structured text without cutting words; fall back to fixed windows."""
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    units, found_structure = _logical_units(text)
    if not found_structure:
        return _split_fixed(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    for unit in units:
        chunks.extend(_split_logical_unit(unit, chunk_size, chunk_overlap))
    return chunks


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "fixed",
) -> list[str]:
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    if strategy == "fixed":
        return _split_fixed(text, chunk_size, chunk_overlap)
    if strategy == "logical":
        return split_logical_text(text, chunk_size, chunk_overlap)
    raise ValueError("chunking strategy must be 'fixed' or 'logical'")


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

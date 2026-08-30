from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextChunk:
    text: str
    metadata: dict


def _load_pdf(path: Path) -> list[TextChunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF ingestion") from exc

    reader = PdfReader(str(path))
    docs: list[TextChunk] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(
            TextChunk(
                text=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": i + 1,
                },
            )
        )
    return docs


def load_documents(raw_dir: str | Path) -> list[TextChunk]:
    root = Path(raw_dir)
    if not root.exists():
        return []

    chunks: list[TextChunk] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            chunks.append(
                TextChunk(
                    text=path.read_text(encoding="utf-8", errors="ignore"),
                    metadata={"source": str(path), "filename": path.name},
                )
            )
        elif suffix == ".pdf":
            chunks.extend(_load_pdf(path))
    return chunks

import pytest

from src.ingestion.chunking import chunk_documents, split_text
from src.ingestion.preprocessing import clean_text
from src.ingestion.document_loader import TextChunk


def test_split_text_with_overlap() -> None:
    text = "abcdefghij"
    chunks = split_text(text, chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_pdf_page_chunk_ids_are_globally_unique() -> None:
    documents = [
        TextChunk("page one", {"source": "guide.pdf", "page": 1}),
        TextChunk("page two", {"source": "guide.pdf", "page": 2}),
    ]
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=0)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == 2


def test_preprocessing_preserves_structural_line_breaks() -> None:
    assert clean_text("# Título\n\nArtículo 1.   Objeto") == "# Título\n\nArtículo 1. Objeto"


def test_split_text_rejects_non_fixed_strategy() -> None:
    with pytest.raises(ValueError, match="must be 'fixed'"):
        split_text("text", chunk_size=10, chunk_overlap=0, strategy="unsupported")

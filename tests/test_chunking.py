from src.ingestion.chunking import chunk_documents, split_text
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

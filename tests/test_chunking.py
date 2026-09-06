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


def test_logical_split_preserves_article_structure_and_words() -> None:
    text = "TÍTULO I\nArtículo 1. Objeto.\nEste Reglamento establece reglas.\n\nArtículo 2. Ámbito.\nSe aplica a sistemas de IA."

    chunks = split_text(text, chunk_size=55, chunk_overlap=0, strategy="logical")

    assert any("Artículo 1." in chunk for chunk in chunks)
    assert any("Artículo 2." in chunk for chunk in chunks)
    assert all(len(chunk) <= 55 or len(chunk.split()) == 1 for chunk in chunks)


def test_logical_split_falls_back_to_fixed_windows_without_structure() -> None:
    assert split_text("abcdefghij", 4, 1, strategy="logical") == ["abcd", "defg", "ghij"]


def test_logical_split_recognizes_numbered_headings() -> None:
    text = "1. Preámbulo\nEl documento establece el marco general.\n\n1.1 Objetivo\nDescribe las obligaciones aplicables."

    chunks = split_text(text, chunk_size=200, chunk_overlap=0, strategy="logical")

    assert len(chunks) == 2
    assert chunks[0].startswith("1. Preámbulo")
    assert chunks[1].startswith("1.1 Objetivo")


def test_logical_split_applies_overlap_between_chunks() -> None:
    text = "Artículo 1. " + " ".join(f"palabra{i}." for i in range(40))

    chunks = split_text(text, chunk_size=100, chunk_overlap=20, strategy="logical")

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert set(chunks[0].split()[-2:]).intersection(chunks[1].split()[:3])


def test_preprocessing_preserves_structural_line_breaks() -> None:
    assert clean_text("# Título\n\nArtículo 1.   Objeto") == "# Título\n\nArtículo 1. Objeto"


def test_logical_split_keeps_list_items_together() -> None:
    text = "Artículo 3. Requisitos:\n- transparencia\n- supervisión humana"

    chunks = split_text(text, chunk_size=80, chunk_overlap=0, strategy="logical")

    assert len(chunks) == 1
    assert "- transparencia" in chunks[0]
    assert "- supervisión humana" in chunks[0]

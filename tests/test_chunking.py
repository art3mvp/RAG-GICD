from src.ingestion.chunking import split_text


def test_split_text_with_overlap() -> None:
    text = "abcdefghij"
    chunks = split_text(text, chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]

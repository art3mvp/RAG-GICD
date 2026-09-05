from src.ingestion.document_loader import TextChunk
from src.retrievers.bm25_retriever import BM25Retriever


def test_bm25_retrieves_relevant_chunk() -> None:
    chunks = [
        TextChunk(text="Python is a programming language", metadata={"source": "a", "chunk_id": "a#0"}),
        TextChunk(text="Cats are friendly animals", metadata={"source": "b", "chunk_id": "b#0"}),
    ]
    retriever = BM25Retriever()
    retriever.build(chunks)

    results = retriever.retrieve("programming language", top_k=1)
    assert results[0].source == "a"


def test_bm25_round_trip(tmp_path) -> None:
    chunks = [
        TextChunk(text="alpha beta", metadata={"source": "a", "chunk_id": "a#0"}),
        TextChunk(text="gamma delta", metadata={"source": "b", "chunk_id": "b#0"}),
    ]
    path = tmp_path / "bm25.pkl"
    retriever = BM25Retriever()
    retriever.build(chunks)
    retriever.save(path)

    loaded = BM25Retriever.load(path)
    assert loaded.is_ready
    assert loaded.retrieve("alpha", top_k=1)[0].metadata["chunk_id"] == "a#0"

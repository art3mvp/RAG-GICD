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

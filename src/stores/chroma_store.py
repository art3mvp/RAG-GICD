from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.document_loader import TextChunk


def _get_chroma_class():
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        return Chroma


class ChromaStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "hybrid_rag") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._store: Any | None = None

    def exists(self) -> bool:
        return self.persist_dir.exists() and any(self.persist_dir.iterdir())

    def build(self, chunks: list[TextChunk], embeddings) -> None:
        Chroma = _get_chroma_class()
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise RuntimeError("langchain-core is required") from exc

        docs = [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name,
        )

    def save(self) -> None:
        if self._store is None:
            raise RuntimeError("No Chroma index built")

    def load(self, embeddings) -> None:
        Chroma = _get_chroma_class()
        self._store = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=embeddings,
        )

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        if self._store is None:
            raise RuntimeError("Chroma store not initialized")
        return self._store.similarity_search_with_relevance_scores(query, k=k)

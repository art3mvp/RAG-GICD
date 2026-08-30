from __future__ import annotations

from pathlib import Path

from src.ingestion.document_loader import TextChunk


class FAISSStore:
    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store = None

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def build(self, chunks: list[TextChunk], embeddings) -> None:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
        except ImportError as exc:
            raise RuntimeError("langchain-community is required for FAISS operations") from exc

        docs = [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
        self._store = FAISS.from_documents(docs, embeddings)

    def save(self) -> None:
        if self._store is None:
            raise RuntimeError("No FAISS index built")
        self._store.save_local(str(self.index_dir))

    def load(self, embeddings) -> None:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise RuntimeError("langchain-community is required for FAISS operations") from exc

        self._store = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def similarity_search_with_score(self, query: str, k: int):
        if self._store is None:
            raise RuntimeError("FAISS store not initialized")
        return self._store.similarity_search_with_score(query, k=k)

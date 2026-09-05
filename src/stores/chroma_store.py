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
    def __init__(self, persist_dir: str | Path, collection_name: str = "dense_rag") -> None:
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

        if self.exists():
            existing = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings,
            )
            existing.delete_collection()

        docs = [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name,
            ids=ids,
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

    def validate(
        self,
        expected_count: int | None = None,
        expected_chunk_ids: set[str] | None = None,
    ) -> None:
        if self._store is None:
            raise RuntimeError("Chroma store not initialized")
        collection = getattr(self._store, "_collection", None)
        if collection is None:
            raise RuntimeError("Chroma collection is unavailable")
        count = collection.count()
        if expected_count is not None and count != expected_count:
            raise RuntimeError(
                f"Chroma collection count mismatch: expected {expected_count}, found {count}"
            )
        if expected_chunk_ids is not None:
            records = collection.get(include=["metadatas"])
            actual_ids = {
                str(metadata.get("chunk_id"))
                for metadata in records.get("metadatas", [])
                if metadata and metadata.get("chunk_id") is not None
            }
            if actual_ids != expected_chunk_ids:
                raise RuntimeError("Chroma collection chunk IDs do not match processed chunks")

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        if self._store is None:
            raise RuntimeError("Chroma store not initialized")
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def similarity_search_with_score(self, query: str, k: int):
        if self._store is None:
            raise RuntimeError("Chroma store not initialized")
        return self._store.similarity_search_with_score(query, k=k)

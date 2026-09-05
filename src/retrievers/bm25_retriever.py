from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.document_loader import TextChunk
from src.retrievers.dense_retriever import RetrievalResult


@dataclass
class BM25Index:
    corpus: list[str]
    chunks: list[TextChunk]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._index = None
        self._chunks: list[TextChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def chunks(self) -> list[TextChunk]:
        return list(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    @property
    def parameters(self) -> tuple[float, float]:
        return self._k1, self._b

    def build(self, chunks: list[TextChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required") from exc

        self._chunks = chunks
        self._tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self._index = BM25Okapi(self._tokenized_corpus, k1=self._k1, b=self._b)

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("BM25 index not built")
        payload = {
            "artifact_version": 1,
            "tokenization_version": 1,
            "k1": self._k1,
            "b": self._b,
            "tokenized_corpus": self._tokenized_corpus,
            "chunks": [{"text": chunk.text, "metadata": chunk.metadata} for chunk in self._chunks],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        """Load a trusted local artifact produced by :meth:`save`."""
        target = Path(path)
        if not target.exists():
            raise RuntimeError(f"BM25 index not found: {target}")
        try:
            with target.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise RuntimeError(f"Unable to load BM25 index: {target}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid BM25 artifact")
        if payload.get("artifact_version") != 1 or payload.get("tokenization_version") != 1:
            raise RuntimeError("Unsupported BM25 artifact version")
        chunks = [
            TextChunk(text=item["text"], metadata=dict(item["metadata"]))
            for item in payload["chunks"]
        ]
        retriever = cls(k1=float(payload["k1"]), b=float(payload["b"]))
        retriever._tokenized_corpus = [list(tokens) for tokens in payload["tokenized_corpus"]]
        retriever._chunks = chunks
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required") from exc
        retriever._index = BM25Okapi(
            retriever._tokenized_corpus, k1=retriever._k1, b=retriever._b
        )
        return retriever

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("BM25 index not loaded or built")

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            metadata = dict(chunk.metadata)
            results.append(
                RetrievalResult(
                    text=chunk.text,
                    source=str(metadata.get("source", "unknown")),
                    score=float(score),
                    retrieval_method="bm25",
                    metadata=metadata,
                )
            )
        return results

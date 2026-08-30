from __future__ import annotations

from dataclasses import dataclass

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

    def build(self, chunks: list[TextChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required") from exc

        self._chunks = chunks
        self._tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self._index = BM25Okapi(self._tokenized_corpus, k1=self._k1, b=self._b)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("BM25 index not built")

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

from __future__ import annotations

from src.retrievers.dense_retriever import RetrievalResult


class BaseReranker:
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results


def get_reranker(enabled: bool) -> BaseReranker:
    if not enabled:
        return NoOpReranker()
    raise NotImplementedError("Reranking is enabled but no reranker implementation is configured")

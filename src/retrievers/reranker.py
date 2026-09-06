from __future__ import annotations

from typing import TYPE_CHECKING

from src.retrievers.dense_retriever import RetrievalResult

if TYPE_CHECKING:
    from src.config.settings import AppSettings


class BaseReranker:
    def load(self) -> None:
        """Load any resources needed before reranking."""
        return None

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, local_files_only=True)
        return self.model

    def load(self) -> None:
        self._get_model()

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [[query, res.text] for res in results]
        scores = self._get_model().predict(pairs)

        for res, score in zip(results, scores):
            res.score = float(score)
            res.retrieval_method = "hybrid_cross_encoder_reranked"

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def get_reranker(settings: AppSettings) -> BaseReranker:
    if not settings.reranker_enabled:
        return NoOpReranker()
    return CrossEncoderReranker(model_name=settings.reranker_model)
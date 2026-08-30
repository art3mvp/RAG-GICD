from __future__ import annotations
from sentence_transformers import CrossEncoder
from src.retrievers.dense_retriever import RetrievalResult

class BaseReranker:
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError

class NoOpReranker(BaseReranker):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        return results[:top_k]

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [[query, res.text] for res in results]
        
        scores = self.model.predict(pairs)

        for res, score in zip(results, scores):
            res.score = float(score)
            res.retrieval_method = "hybrid_cross_encoder_reranked"

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

def get_reranker(settings: AppSettings) -> BaseReranker:
    if not settings.reranker_enabled:
        return NoOpReranker()
    return CrossEncoderReranker(model_name=settings.reranker_model)